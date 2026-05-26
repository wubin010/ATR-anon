"""The test-session stage: single-shot TestSession generation (no QC, no refine loop).

`generate_candidate(rule, persona_id, model)` calls the LLM once and returns
a stamped session dict (or None on LLM error). QC + refine loop lives in
`pipeline.py`.

Persona is intentionally NOT in the prompt — the test session is rule-driven
and the runner's test-phase agent has no persona visibility either. Few-shot
exemplars from the curated pool do the heavy lifting.

Also exposes helpers reused by pipeline + refine:
  - `domain_of_rule(rule)`        → ontology reverse-lookup
  - `stamp_session(session, rule, persona_id, domain)`  → attach identity / rule_ref
"""
from __future__ import annotations

import logging
from pathlib import Path

from datagen._common import (
    DOMAINS,
    base_tool_names,
    domain_of_rule,
    domain_tools,
    fill_prompt,
    load_prompt,
    render_domain_tools,
    render_ref_schema,
    rule_is_permission,
)
from datagen.test_sessions.pool import load_pool_exemplars, render_exemplars_md
from lib.llm import GPT, call_llm_json  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"

logger = logging.getLogger(__name__)


class CandidateGenerationInfraError(RuntimeError):
    """Raised when candidate generation fails due to the LLM provider."""


# ---------------------------------------------------------------------------
# Helpers (public — pipeline + refine import from here)
# ---------------------------------------------------------------------------


def rule_needs_base(rule: dict) -> bool:
    """True iff the rule's gold tool is `get_user_confirmation` (a base-
    domain tool). `stamp_session` uses this to decide whether base tools
    should appear in the agent's action space — we only expose them for
    rules whose gold actually calls them.
    """
    return rule_is_permission(rule)


def derive_required_actions(
    rule: dict, gold_value,
) -> list[dict]:
    """Mechanically generate labels.task_success.required_actions from
    rule + gold_value (one value, see table below).

    Dispatched by rule.check_type — four flat branches:

      tool_identity (mutate, action_step.param=null):
        gold_value: null
        → {tool: action_step.tool, arguments: {}, compare_args: null}

      param_id (mutate):
        gold_value:
          - "<ref_id>"       when param is `string` (single id)
          - ["<ref_id>", …]  when param is `array[string]` (list of ids)
        → {tool, arguments: {param: gold_value}, compare_args: [param]}

      param_enum (mutate):
        gold_value: "<enum_value>"
        → {tool, arguments: {param: gold_value}, compare_args: [param]}

      confirm:
        gold_value:
          - null              (confirm-only — no inner-param preference)
          - {<inner_param>: <inner_value>}  (confirm + inner-param preference)
        action_step.tool MUST be "get_user_confirmation"; action_step.param
        is the inner mutate tool name (used as target_tool).

        confirm-only:
          → {tool: "get_user_confirmation",
             arguments: {target_tool: <inner mutate>},
             compare_args: ["target_tool"]}

        confirm + inner param:
          → {tool: "get_user_confirmation",
             arguments: {target_tool: <inner mutate>, target_params: <gold_value>},
             compare_args: ["target_tool", "target_params"]}

    Returns [] on malformed rule (caller treats as gen failure).
    """
    step = rule.get("action_step") or {}
    if not isinstance(step, dict):
        return []
    tool = step.get("tool")
    param = step.get("param")
    check_type = rule.get("check_type")

    if check_type == "confirm":
        if tool != "get_user_confirmation":
            return []
        target_tool = param  # inner mutate tool name
        if not target_tool:
            return []
        if gold_value is None:
            return [{
                "tool": "get_user_confirmation",
                "arguments": {"target_tool": target_tool},
                "compare_args": ["target_tool"],
            }]
        if not isinstance(gold_value, dict):
            return []
        return [{
            "tool": "get_user_confirmation",
            "arguments": {
                "target_tool": target_tool,
                "target_params": gold_value,
            },
            "compare_args": ["target_tool", "target_params"],
        }]

    # tool_identity / param_id / param_enum (mutate rule)
    if param:
        # Defense-in-depth: a param-typed rule with gold_value=None would
        # produce a RequiredAction whose compare_args names a key absent
        # from arguments — evaluator would then compare None vs actual and
        # always fail. static_check catches this upstream, but emit []
        # here so callers that bypass static_check fail fast.
        if gold_value is None:
            return []
        action: dict = {
            "tool": tool,
            "arguments": {param: gold_value},
            "compare_args": [param],
        }
        return [action]
    return [{
        "tool": tool,
        "arguments": {},
        "compare_args": None,
    }]


def stamp_session(session: dict, rule: dict, persona_id: str, domain: str) -> dict:
    """Attach identity + rule_ref + derived labels. Idempotent.

    LLM emits `gold_value` (single field, see derive_required_actions for
    shape per check_type). This function mechanically derives the full
    `labels.task_success.required_actions` from rule + gold_value —
    eliminating LLM-authored label errors (wrong tool / mistyped
    compare_args / hallucinated ids in arguments).

    Also overrides local_env.tools with the whole-domain tool list — we
    expose every tool in the domain to the agent (both learning and test)
    so the agent has more "preference anchors" to reason about.
    References stay whatever the LLM authored.
    """
    rule_id = rule["rule_id"]
    session = dict(session)
    # Drop fields outside the schema that the LLM might have copied from
    # few-shot data or cached output. Schema is {instruction, gold_value, references}.
    session.pop("hidden_rule_slots", None)
    session.pop("gold_markers", None)
    session["session_id"] = f"{rule_id}_test"
    session["session_type"] = "test"
    session["domain"] = domain
    session["rule_id"] = rule_id
    session["rule_ref"] = {
        "rule_id": rule_id,
        "rule_text": rule["rule_text"],
        "canonical_answer": rule["canonical_answer"],
    }
    # Action-space: full domain tool set; only include base tools
    # (get_user_confirmation) when the rule's gold trajectory actually
    # calls them. Avoids polluting action space for rules that don't
    # need confirmation prologues.
    domain_tool_names = [t["name"] for t in domain_tools(domain)]
    if rule_needs_base(rule):
        domain_tool_names.extend(sorted(base_tool_names()))
    local_env = session.setdefault("local_env", {})
    local_env["tools"] = domain_tool_names
    # LLM emits `references` at top level under the new 3-field output
    # schema; pull it into local_env.references for downstream consumers.
    if "references" in session and "references" not in local_env:
        local_env["references"] = session.pop("references")
    local_env.setdefault("references", [])

    # Derive labels from gold_value. KEY MUST be present even when the
    # value is null (e.g. tool_identity / confirm-only) so we use
    # `in` rather than truthiness check.
    if "gold_value" in session:
        gv = session["gold_value"]
        required = derive_required_actions(rule, gv)
        session.setdefault("labels", {}).setdefault("task_success", {})[
            "required_actions"
        ] = required

    return session


# ---------------------------------------------------------------------------
# LLM call — single attempt, stamped
# ---------------------------------------------------------------------------


def generate_candidate(
    rule: dict,
    persona_id: str,
    model: str = GPT,
) -> dict | None:
    """Generate one candidate TestSession. Returns stamped dict or None on
    local validation/domain resolution failure. LLM provider failures raise
    CandidateGenerationInfraError so the pipeline can fail the stage instead
    of counting infrastructure downtime as a data-quality drop.

    Does NOT run static_check / QC / refine — caller (pipeline) handles
    validation and refine loop.
    """
    domain = domain_of_rule(rule)
    if domain not in DOMAINS:
        logger.warning("rule %s: cannot resolve domain (tool=%s)",
                       rule.get("rule_id"),
                       (rule.get("action_step") or {}).get("tool"))
        return None

    template = load_prompt(PROMPTS_DIR, name="gen")
    exemplars = load_pool_exemplars(rule, max_count=5)
    # Strip stage-local diagnostic fields before showing to the gen LLM.
    # _qc (tag QC verdict) could bias gen output if leaked through.
    rule_for_prompt = {
        k: v for k, v in rule.items() if k != "_qc"
    }
    prompt = fill_prompt(
        template,
        RULE_JSON=rule_for_prompt,
        DOMAIN=domain,
        DOMAIN_TOOLS=render_domain_tools(domain),
        REF_SCHEMA=render_ref_schema(domain),
        FEW_SHOT_EXAMPLES=render_exemplars_md(exemplars),
    )

    try:
        # reasoning_effort="on": plan-then-generate asks the model to
        # internally do 5-step CoT (dissect rule → refs → gold pick →
        # instruction → self-check). There's no refine loop anymore so
        # first output has to be good — worth the extra thinking tokens.
        result = call_llm_json(
            prompt, model=model, temperature=0.5, max_retries=3,
            reasoning_effort="on",
        )
    except Exception as e:
        logger.warning("generate_candidate LLM failed for %s: %s", rule.get("rule_id"), e)
        raise CandidateGenerationInfraError(
            f"generate_candidate LLM failed for {rule.get('rule_id')}: {e}"
        ) from e

    if not isinstance(result, dict):
        logger.warning("generate_candidate non-dict for %s: %s",
                       rule.get("rule_id"), type(result).__name__)
        return None

    return stamp_session(result, rule, persona_id, domain)
