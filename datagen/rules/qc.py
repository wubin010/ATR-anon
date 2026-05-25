"""Rule QC — final gate after generation.

Reads rules_gen.json (bound rules from rule generation). One LLM call per persona —
batch tag QC on two independent per-rule axes:

  · counter_default  (yes | no)  + reason
  · binding_sound    (yes | no)  + reason

A rule is kept iff counter_default == yes AND binding_sound == yes.
The tag is computed in code, not by the LLM.

Cross-rule duplicates and contradictions are NOT handled here — gen_prompt.md
has hard rules forbidding duplicate / opposite bindings, and session_gen's
env design (multi-attribute candidate refs) satisfies multiple non-conflicting
same-tool rules in one task.

Few-shot pools (one JSON per axis at data/qc_pool/):
  - counter_default_pool.json
  - binding_sound_pool.json
Each entry: {rule_text, counterfactual_default, check_type, action_step,
verdict, label, explanation}. verdict=no entries are rendered as ✗ fail
examples; verdict=yes are rendered as ✓ guards (rule looks risky to
this axis but actually passes — anti-false-positive).

Pipeline: gen (rule generation) → qc (rule QC, this file).

Input:  data/personas/<uuid>/rules_gen.json
Output: data/personas/<uuid>/rules.json              (final — downstream reads this)
        data/personas/<uuid>/rules_qc_dropped.json
"""
from __future__ import annotations

import logging
from pathlib import Path

from datagen._common import (
    PERSONAS_DIR,
    fill_prompt,
    load_prompt,
    read_json,
    to_json,
    write_json,
)
from lib.llm import GPT, call_llm_json  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
DEFAULT_MODEL = GPT
_QC_POOL_DIR = HERE.parent.parent / "data" / "few_shot_pool" / "rules_qc"

logger = logging.getLogger(__name__)


def _load_pool(filename: str) -> str:
    """Render a per-axis pool with both ✗ fails and ✓ guards.

    Each entry: {rule_text, counterfactual_default, check_type,
    action_step, verdict, label, explanation}.
      · verdict=no  → ✗ fail example (rule should fail this axis)
      · verdict=yes → ✓ guard example (rule looks risky but passes)
    """
    path = _QC_POOL_DIR / filename
    if not path.is_file():
        return "(pool empty)"
    examples = read_json(path) or []
    blocks_no: list[str] = []
    blocks_yes: list[str] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        rule = ex.get("rule_text", "")
        cf = ex.get("counterfactual_default", "")
        ct = ex.get("check_type", "")
        step = ex.get("action_step") or {}
        tool = step.get("tool", "") if isinstance(step, dict) else ""
        param = step.get("param") if isinstance(step, dict) else None
        verdict = ex.get("verdict", "")
        label = ex.get("label", "")
        explanation = ex.get("explanation", "")
        action_str = tool + (f"(param={param})" if param else "")
        marker = "✗" if verdict == "no" else "✓"
        block = (
            f"{marker} rule: \"{rule}\"\n"
            f"  counterfactual: \"{cf}\"\n"
            f"  check_type: {ct}  action: {action_str}\n"
            f"  label: {label}\n"
            f"  why: {explanation}"
        )
        if verdict == "no":
            blocks_no.append(block)
        else:
            blocks_yes.append(block)

    sections: list[str] = []
    if blocks_no:
        sections.append(
            "**✗ examples — these FAIL this axis:**\n\n"
            + "\n\n".join(blocks_no)
        )
    if blocks_yes:
        sections.append(
            "**✓ examples — these PASS this axis (don't mistake for fail):**\n\n"
            + "\n\n".join(blocks_yes)
        )
    return "\n\n".join(sections) if sections else "(pool empty)"


def _annotate_drop(rule: dict, reason: str) -> dict:
    """Annotate a rule with the full _qc shape on a code-side drop path
    (LLM didn't tag, returned non-list, etc.). Keeps downstream readers
    of rules_qc_dropped.json able to assume _qc is always present.
    """
    r = dict(rule)
    r["_qc"] = {
        "counter_default": "no",
        "counter_default_reason": reason,
        "binding_sound": "no",
        "binding_sound_reason": reason,
        "tag": "remove",
    }
    return r


def qc_persona(
    persona_id: str,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], list[dict]]:
    """Single-LLM-call two-axis tag QC. Returns (kept, dropped).

    A rule is kept iff counter_default == yes AND binding_sound == yes.
    The tag is computed here from the LLM's per-axis labels.
    """
    pdir = PERSONAS_DIR / persona_id
    gen_path = pdir / "rules_gen.json"
    if not gen_path.exists():
        logger.warning("[%s] no rules_gen.json, skip", persona_id)
        return [], []

    rules = read_json(gen_path)
    if not isinstance(rules, list):
        logger.error("[%s] rules_gen.json is %s, not list — skipping",
                     persona_id, type(rules).__name__)
        return [], []

    template = load_prompt(PROMPTS_DIR, name="qc")
    prompt = fill_prompt(
        template,
        RULES_JSON=to_json(rules),
        COUNTER_DEFAULT_POOL=_load_pool("counter_default_pool.json"),
        BINDING_SOUND_POOL=_load_pool("binding_sound_pool.json"),
    )

    logger.info("[%s] qc: %d bound rules...", persona_id, len(rules))
    result = call_llm_json(prompt, model=model, temperature=0.1, max_retries=3)

    if not isinstance(result, list):
        logger.error("[%s] qc: LLM returned non-list: %s",
                     persona_id, type(result).__name__)
        return [], [_annotate_drop(r, "llm_returned_non_list") for r in rules]

    # Match on rule_text (rule_id is stamped by gen).
    tags_by_text: dict[str, dict] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        rt = item.get("rule_text", "")
        if rt in tags_by_text:
            logger.warning(
                "[%s] qc: LLM returned duplicate rule_text %r — keeping last; "
                "gen-stage dedup should have prevented this",
                persona_id, rt[:60],
            )
        tags_by_text[rt] = item

    kept: list[dict] = []
    dropped: list[dict] = []
    for rule in rules:
        rule_text = rule.get("rule_text", "")

        tag_item = tags_by_text.get(rule_text)
        if tag_item is None:
            logger.warning(
                "[%s] qc: no tag for rule %r, defaulting to drop",
                persona_id, (rule_text or "")[:60],
            )
            dropped.append(_annotate_drop(rule, "llm_did_not_tag_this_rule"))
            continue

        cd = tag_item.get("counter_default", "")
        cd_reason = tag_item.get("counter_default_reason", "")
        bs = tag_item.get("binding_sound", "")
        bs_reason = tag_item.get("binding_sound_reason", "")
        passed = (cd == "yes" and bs == "yes")
        tag = "keep" if passed else "remove"

        annotated = dict(rule)
        annotated["_qc"] = {
            "counter_default": cd,
            "counter_default_reason": cd_reason,
            "binding_sound": bs,
            "binding_sound_reason": bs_reason,
            "tag": tag,
        }
        if passed:
            kept.append(annotated)
        else:
            dropped.append(annotated)

    logger.info("[%s] qc: %d kept, %d dropped", persona_id, len(kept), len(dropped))
    return kept, dropped


def run_persona(
    persona_id: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> dict:
    pdir = PERSONAS_DIR / persona_id
    kept_path = pdir / "rules.json"
    if kept_path.exists() and not force:
        logger.info("[%s] rules.json exists, skip (use force=True)", persona_id)
        return {"persona_id": persona_id, "status": "skipped"}

    kept, dropped = qc_persona(persona_id, model)
    if kept:
        write_json(kept_path, kept)
    if dropped:
        write_json(pdir / "rules_qc_dropped.json", dropped)
    return {
        "persona_id": persona_id,
        "status": "ok",
        "kept": len(kept),
        "dropped": len(dropped),
    }
