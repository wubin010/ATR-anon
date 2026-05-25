"""Rule generation.

One LLM call per persona produces a list of bound rules (each carrying
check_type + action_step). No "unbindable" output — if a candidate rule
doesn't fit any check_type cleanly, the LLM simply skips it and emits
fewer rules.

Pipeline: gen (this file, rule generation) → qc (rule QC).

Input:  data/personas/<uuid>/structured.json
Output: data/personas/<uuid>/rules_gen.json
        data/personas/<uuid>/rules_gen_rejected.json  (validator rejected)

Validation per rule:
  recall fields  rule_text / canonical_answer / counterfactual_default /
                 evidence
  binding        check_type / action_step (a single dict, not a list)
  ontology       tool exists; tool's domain is in DOMAINS, OR tool ==
                 get_user_confirmation (the special confirm tool)
  decision tool  not a read-only exploration prefix (search_/list_/lookup_/
                 get_/view_/browse_/compare_/shortlist_/filter_/rank_/
                 refine_/narrow_/check_/find_/track_/review_);
                 get_user_confirmation is the one `get_`-prefixed exception
  confirm rules  tool == get_user_confirmation:
                   - check_type == tool_identity
                   - param names a real mutate tool (not read-prefix,
                     in DOMAINS, not get_user_confirmation itself)
  mutate rules   tool != get_user_confirmation:
                   - check_type ↔ param compat:
                     tool_identity → param is null
                     param_id      → param ends with _id or _ids
                     param_enum    → param's ontology type starts with enum[
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from datagen._common import (
    DOMAINS,
    PERSONAS_DIR,
    fill_prompt,
    format_structured_persona,
    load_prompt,
    read_json,
    render_bindable_tools,
    tool_map,
    write_json,
)
from datagen.config import CONFIG
from lib.llm import GPT, call_llm_json  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
DEFAULT_MODEL = GPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_EVIDENCE_TYPES = {"direct", "inferred"}
_VALID_CHECK_TYPES = {"tool_identity", "param_id", "param_enum", "confirm"}

_RECALL_FIELDS = (
    "rule_text",
    "canonical_answer",
    "counterfactual_default",
    "evidence",
)

# Read-prefix tools cannot be the gold tool of a rule. See
# datagen/_common.py:_RULE_NON_BINDABLE_PREFIXES for rationale (hard reads
# trivially-hit-gold; monitoring/checking enum-rules always trip
# framework_blind_spot:A on the catch-all "all" enum value).
# `get_user_confirmation` is the one `get_`-prefixed exception.
_EXPLORATION_TOOL_PREFIXES = (
    "search_", "list_", "lookup_", "get_", "view_", "browse_", "compare_",
    "shortlist_", "filter_", "rank_", "refine_", "narrow_",
    "check_", "find_", "track_", "review_",
)
_CONFIRM_TOOL = "get_user_confirmation"

# Validator double-safety: even though `render_bindable_tools()` already
# hides runtime-only tools from the LLM, keep the validator check so a
# stray LLM hallucination (proposing a tool not in the rendered list)
# can't slip into rules_gen.json. Single source of truth lookup via `_common`.
from datagen._common import runtime_entity_tools as _get_runtime_entity_tools


def _validate_recall_fields(rule: dict) -> list[str]:
    errors: list[str] = []
    for field in _RECALL_FIELDS:
        v = rule.get(field)
        if v is None or (isinstance(v, str) and not v.strip()) or (
            isinstance(v, list) and not v
        ):
            errors.append(f"missing_or_empty:{field}")
    if errors:
        return errors

    ev = rule["evidence"]
    if not isinstance(ev, list) or not ev:
        errors.append("evidence_not_list_or_empty")
    else:
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                errors.append(f"evidence_{i}_not_dict")
                continue
            if e.get("type") not in _VALID_EVIDENCE_TYPES:
                errors.append(f"evidence_{i}_invalid_type:{e.get('type')}")
            content = e.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"evidence_{i}_empty_content")
    return errors


def _validate_binding_fields(rule: dict, tmap: dict[str, dict]) -> list[str]:
    """Validate a bindable rule's check_type + action_step.

    Four check_type branches dispatched independently:
      tool_identity / param_id / param_enum (mutate rule):
        - action_step.tool is a mutate tool (not a read-prefix, in DOMAINS)
        - check_type ↔ param compatibility:
            tool_identity → param is None
            param_id      → param ends with _id or _ids
            param_enum    → param's ontology type starts with `enum[`
      confirm (wrapper):
        - action_step.tool MUST be `get_user_confirmation`
        - action_step.param names the inner mutate tool — must exist in
          tmap, be in DOMAINS, not a read-prefix, not get_user_confirmation
          itself
    """
    errors: list[str] = []

    check_type = rule.get("check_type")
    if not isinstance(check_type, str) or not check_type:
        errors.append("check_type_missing_or_not_string")
    elif check_type not in _VALID_CHECK_TYPES:
        errors.append(f"invalid_check_type:{check_type}")

    step = rule.get("action_step")
    if not isinstance(step, dict):
        errors.append("action_step_not_dict_or_missing")
        return errors

    tool = step.get("tool")
    if not tool or tool not in tmap:
        errors.append(f"action_step_unknown_tool:{tool}")
        return errors

    param = step.get("param")
    if param is not None and not isinstance(param, str):
        errors.append(f"param_must_be_string_or_null:got={type(param).__name__}")
        return errors

    if check_type == "confirm":
        if tool != _CONFIRM_TOOL:
            errors.append(
                f"confirm_check_type_requires_get_user_confirmation_tool:got={tool}"
            )
        if not param:
            errors.append(
                "confirm_rule_param_must_name_target_mutate_tool:got_null"
            )
        elif param not in tmap:
            errors.append(f"confirm_target_unknown_tool:{param}")
        elif param == _CONFIRM_TOOL:
            errors.append("confirm_target_is_get_user_confirmation_self_loop")
        elif param.startswith(_EXPLORATION_TOOL_PREFIXES):
            errors.append(f"confirm_target_is_exploration_prefix:{param}")
        elif param in _get_runtime_entity_tools():
            errors.append(
                f"confirm_target_is_runtime_only_entity:{param} "
                "(no mechanism to pre-hydrate this entity from references)"
            )
        elif tmap[param].get("domain") not in DOMAINS:
            errors.append(
                f"confirm_target_domain_unregistered:{param}:"
                f"{tmap[param].get('domain')}"
            )
        return errors

    # Reject confirm tool in mutate-shaped rules (check_type ≠ confirm).
    if tool == _CONFIRM_TOOL:
        errors.append(
            f"get_user_confirmation_requires_check_type_confirm:got={check_type}"
        )
        return errors

    # ── mutate rule (tool_identity / param_id / param_enum) ───────────
    if tool.startswith(_EXPLORATION_TOOL_PREFIXES):
        errors.append(f"action_step_tool_is_exploration_prefix:{tool}")
    if tool in _get_runtime_entity_tools():
        errors.append(
            f"action_step_tool_is_runtime_only_entity:{tool} "
            "(no mechanism to pre-hydrate this entity from references)"
        )
    if tmap[tool].get("domain") not in DOMAINS:
        errors.append(
            f"action_step_tool_domain_unregistered:{tool}:"
            f"{tmap[tool].get('domain')}"
        )

    tool_params = {p["name"]: p for p in tmap[tool]["parameters"]}

    if check_type == "tool_identity":
        if param is not None:
            errors.append(f"tool_identity_must_have_null_param:got={param!r}")
    elif check_type == "param_id":
        if not param:
            errors.append("param_id_must_have_param")
        elif param not in tool_params:
            errors.append(f"param_not_on_tool:{tool}.{param}")
        elif not (param.endswith("_id") or param.endswith("_ids")):
            errors.append(
                f"param_id_param_must_end_with_id_or_ids:got={param!r}"
            )
    elif check_type == "param_enum":
        if not param:
            errors.append("param_enum_must_have_param")
        elif param not in tool_params:
            errors.append(f"param_not_on_tool:{tool}.{param}")
        else:
            ptype = (tool_params[param].get("type") or "")
            if not ptype.startswith("enum["):
                errors.append(
                    f"param_enum_param_must_be_enum_typed:got={param}:type={ptype}"
                )

    return errors


def _validate_rule(rule: dict, tmap: dict[str, dict]) -> list[str]:
    """Full validation. Empty list = pass."""
    errors = _validate_recall_fields(rule)
    if errors:
        return errors
    errors.extend(_validate_binding_fields(rule, tmap))
    return errors


# ---------------------------------------------------------------------------
# rule_id stamping
# ---------------------------------------------------------------------------

def _stamp_rule_ids(
    rules: list[dict], persona_id: str, tmap: dict[str, dict],
) -> list[dict]:
    """Stamp rule_id using the rule's business domain. Confirm rule's
    domain comes from action_step.param (the would-be mutate tool);
    mutate rule's domain is action_step.tool's own domain.
    """
    counters: dict[str, int] = {}
    out: list[dict] = []
    for r in rules:
        step = r.get("action_step") or {}
        domain = "misc"
        if isinstance(step, dict):
            t = step.get("tool", "")
            target = step.get("param") if t == _CONFIRM_TOOL else t
            if target:
                d = tmap.get(target, {}).get("domain")
                if d and d != "base":
                    domain = d
        counters.setdefault(domain, 0)
        counters[domain] += 1
        r = dict(r)
        r["rule_id"] = f"{persona_id}_{domain}_{counters[domain]:02d}"
        r["persona_id"] = persona_id
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def gen_rules(
    persona_id: str,
    structured: dict,
    n: int,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], list[dict]]:
    """One LLM call per persona. Returns (accepted, rejected).

    `accepted` = validator-passed bound rules.
    `rejected` = validator rejected (malformed shape / hallucinated fields).
    """
    template = load_prompt(PROMPTS_DIR, name="gen")
    prompt = fill_prompt(
        template,
        N_RULES=str(n),
        STRUCTURED_PERSONA=format_structured_persona(structured),
        TOOLS_TEXT=render_bindable_tools(),
    )
    logger.info("[%s] gen: calling LLM (n=%d)...", persona_id, n)
    t0 = time.time()
    result = call_llm_json(prompt, model=model, temperature=0.5, max_retries=3)
    elapsed = time.time() - t0
    logger.info(
        "[%s] gen: LLM returned in %.1fs (%s items)",
        persona_id, elapsed,
        len(result) if isinstance(result, list) else "non-list",
    )

    if not isinstance(result, list):
        logger.warning(
            "[%s] gen: LLM returned %s, expected list — skipping",
            persona_id, type(result).__name__,
        )
        return [], []

    tmap = tool_map()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for rule in result:
        if not isinstance(rule, dict):
            continue
        reasons = _validate_rule(rule, tmap)
        if reasons:
            rule["_reject_reasons"] = reasons
            rejected.append(rule)
        else:
            accepted.append(rule)

    accepted = _stamp_rule_ids(accepted, persona_id, tmap)
    return accepted, rejected


# ---------------------------------------------------------------------------
# Per-persona driver
# ---------------------------------------------------------------------------

def run_persona(
    persona_id: str,
    n: int = CONFIG.rule_count_per_persona,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> dict:
    pdir = PERSONAS_DIR / persona_id
    out_path = pdir / "rules_gen.json"
    if out_path.exists() and not force:
        logger.info("[%s] rules_gen.json exists, skip (use force=True)", persona_id)
        return {"persona_id": persona_id, "status": "skipped"}

    structured_path = pdir / "structured.json"
    if not structured_path.exists():
        logger.error("[%s] missing structured.json — run ingest first", persona_id)
        return {"persona_id": persona_id, "status": "missing_structured"}

    structured = read_json(structured_path)
    accepted, rejected = gen_rules(persona_id, structured, n, model)

    if accepted:
        write_json(out_path, accepted)
    if rejected:
        write_json(pdir / "rules_gen_rejected.json", rejected)

    logger.info(
        "[%s] gen done: %d accepted, %d rejected",
        persona_id, len(accepted), len(rejected),
    )
    return {
        "persona_id": persona_id,
        "status": "ok",
        "accepted": len(accepted),
        "rejected": len(rejected),
    }
