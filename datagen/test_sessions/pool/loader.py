"""Few-shot exemplar loader for test_session_gen.

`load_pool_exemplars(rule, max_count, require_qc_passed)` returns a list of
`{rule, session}` dicts pulled from `data/test_session_pool/`.

Selection strategy:
  1. Build a target signature from the rule: (check_type, is_permission).
     Only exemplars whose signature matches exactly are considered —
     different-signature samples teach the LLM wrong instruction / decoy
     patterns, so cross-category contamination is hard-filtered out (no
     fallback padding).
  2. **Quality gate**: only consider samples whose latest `qc_pool_results_*.json`
     entry has `passed=True` (full-QC-pass). Samples that never cleared QC are
     excluded so they can't contaminate few-shot generation. If no QC results
     file exists yet (fresh repo), all samples are considered.
  3. Within same-signature subset, rank by domain affinity:
       same domain as `prefer_domain`  → score 2
       otherwise                       → score 1
     Ties broken by stable rule_id sort.
  4. Take top `max_count`. If no same-signature exemplar exists, return [];
     callers should treat this as a legitimate "no few-shot for this signature"
     state and let the gen LLM rely on the prompt + rule description alone.

Caller renders with `render_exemplars_md(exemplars)` and substitutes
`{{FEW_SHOT_EXAMPLES}}` in the gen / refine prompt template. The rendered
form shows `gold_markers` (reverse-derived from session.required_actions)
instead of the full `labels` block, so few-shots teach the post-label-
programmatisation schema the new gen prompt expects.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from datagen._common import (
    PROJECT_ROOT,
    domain_of_rule,
    read_json,
    rule_is_permission,
)

POOL_DIR = PROJECT_ROOT / "data" / "few_shot_pool" / "test_session_gen"

logger = logging.getLogger(__name__)



@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict], dict[str, dict]]:
    index = read_json(POOL_DIR / "index.json")
    rules = read_json(POOL_DIR / "rules.json")
    rules_by_id = {r["rule_id"]: r for r in rules}
    return index, rules_by_id


@lru_cache(maxsize=1)
def _qc_passed_rule_ids() -> frozenset[str] | None:
    """Return the set of rule_ids considered QC-passed. Sources are
    unioned — a rule counts as passed if ANY source asserts it:

      1. `index.json` entry has `full_qc_passed=True` (used by the
         promote-to-pool workflow: a session that passed QC in a
         `pipeline.py` run is marked True when moved into the pool)
      2. Any `qc_pool_results_*.json` under POOL_DIR has an entry with
         `passed=True` (used by the `qc_pool` batch-verification
         workflow on pool samples themselves)

    Returns None only when NEITHER source exists / is readable — in
    which case callers skip QC filtering (fresh repo case).

    Multiple qc_pool_results_*.json files: later-mtime files override
    earlier ones per rule_id, so the newest verdict wins.
    """
    # Source 1: index.json full_qc_passed field
    pass_from_index: set[str] = set()
    try:
        index = json.loads((POOL_DIR / "index.json").read_text())
        for e in index:
            if e.get("full_qc_passed") is True:
                pass_from_index.add(e["rule_id"])
    except Exception as e:
        logger.warning("pool loader: can't read index.json for qc: %s", e)

    # Source 2: qc_pool_results_*.json scanner
    pass_from_results: set[str] = set()
    files = sorted(
        POOL_DIR.glob("qc_pool_results*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if files:
        latest_verdict: dict[str, bool] = {}
        for f in files:  # oldest first, later files overwrite
            try:
                data = json.loads(f.read_text())
            except Exception as e:
                logger.warning("skip unreadable qc result %s: %s", f.name, e)
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                rid = entry.get("rule_id")
                if not isinstance(rid, str):
                    continue
                # Treat missing `passed` / errors / static_errors as "not passed"
                latest_verdict[rid] = bool(entry.get("passed"))
        pass_from_results = {rid for rid, ok in latest_verdict.items() if ok}

    union = pass_from_index | pass_from_results
    if not union and not files and not pass_from_index:
        return None  # no source at all — caller should skip filter
    return frozenset(union)


def _derive_gold_value(rule: dict, session: dict):
    """Reverse-derive `gold_value` from an on-disk session's labels +
    rule. Used for rendering few-shot in the post-rewrite schema so the
    gen LLM learns to emit `gold_value` (single field) directly.

    Dispatched by `rule.check_type` (4 flat branches), mirroring
    `gen.derive_required_actions`:

      tool_identity (mutate, action_step.param=null) → null
      param_id   → required_actions[0].arguments[<param>]   # ref id string
      param_enum → required_actions[0].arguments[<param>]   # enum value
      confirm    → null (confirm-only)                      # compare_args=["target_tool"]
                   OR required_actions[0].arguments["target_params"]  # dict
    """
    sess_actions = (
        session.get("labels", {})
        .get("task_success", {})
        .get("required_actions", [])
    ) or []
    if not sess_actions:
        return None
    args = (sess_actions[0].get("arguments") or {})
    check_type = rule.get("check_type")

    if check_type == "confirm" or rule.get("action_step", {}).get("tool") == "get_user_confirmation":
        tp = args.get("target_params")
        if isinstance(tp, dict) and tp:
            return tp
        return None

    step = rule.get("action_step") or {}
    param = step.get("param") if isinstance(step, dict) else None
    if param and param in args:
        return args[param]
    return None


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def _rule_signature(rule: dict) -> tuple:
    """Structural category of a rule. Two rules with the same signature
    share the same instruction-writing / decoy-design pattern; rules with
    different signatures do NOT teach each other and must never cross-
    pollinate as few-shot exemplars.

    Signature = check_type alone, now that confirm is a first-class
    check_type (no need for the auxiliary is_permission boolean).
    """
    # Normalize: a rule with action_step.tool=get_user_confirmation
    # but check_type=tool_identity is a confirm shape — treat as
    # confirm for signature-matching purposes.
    ct = rule.get("check_type")
    if ct == "tool_identity" and rule_is_permission(rule):
        ct = "confirm"
    return (ct,)


def load_pool_exemplars(
    rule: dict,
    max_count: int = 3,
    require_qc_passed: bool = True,
) -> list[dict]:
    """Return ≤max_count exemplars with STRICT signature match to `rule`.

    Hard filter:
      - Only exemplars with the exact same `_rule_signature` as `rule`
        are considered. Different-signature samples teach the LLM
        wrong instruction / decoy patterns — we'd rather have 0 or 1
        few-shot than mix categories.

    Within the same-signature subset:
      - Same domain as `rule` is preferred (tie-break, not filter).
      - QC-pass is enforced by default (`require_qc_passed=True`).

    Returns [] when no same-signature sample exists (caller should treat
    this as a legitimate "no few-shot for this signature" state — the
    gen LLM then relies purely on the rule description + prompt rules).

    Each exemplar dict: `{"rule", "session", "bucket", "domain"}`.
    """
    if max_count <= 0:
        return []

    index, rules_by_id = _load_index()

    # ── QC quality gate ────────────────────────────────────────────────
    if require_qc_passed:
        passed = _qc_passed_rule_ids()
        if passed is None:
            logger.warning(
                "pool loader: no qc_pool_results*.json found — skipping "
                "QC-pass filter (few-shot may include unverified samples)"
            )
        else:
            before = len(index)
            index = [e for e in index if e["rule_id"] in passed]
            logger.debug(
                "pool loader: QC filter kept %d/%d samples", len(index), before,
            )

    # ── Signature hard filter ──────────────────────────────────────────
    target_sig = _rule_signature(rule)
    target_domain = domain_of_rule(rule)
    target_rule_id = rule.get("rule_id")

    candidates: list[dict] = []
    for e in index:
        if e["rule_id"] == target_rule_id:
            continue   # don't self-reference — LLM already has target rule
        entry_rule = rules_by_id.get(e["rule_id"])
        if entry_rule is None:
            continue
        if _rule_signature(entry_rule) != target_sig:
            continue   # different category — never use as few-shot
        candidates.append(e)

    if not candidates:
        logger.info(
            "pool loader: no same-signature exemplars for signature=%s "
            "(domain=%s) — returning empty few-shot list",
            target_sig, target_domain,
        )
        return []

    # ── Domain preference (tie-break within signature) ─────────────────
    def _score(entry: dict) -> int:
        return 2 if (target_domain and entry["domain"] == target_domain) else 1

    candidates.sort(key=lambda e: (-_score(e), e["rule_id"]))
    picks = candidates[:max_count]

    # Hydrate
    out: list[dict] = []
    for e in picks:
        rid = e["rule_id"]
        rule_obj = rules_by_id.get(rid)
        session_path = PROJECT_ROOT / e["path"]
        if rule_obj is None or not session_path.exists():
            continue
        session = read_json(session_path)
        out.append({
            "rule": rule_obj,
            "session": session,
            "bucket": e["bucket"],
            "domain": e["domain"],
        })
    return out


# ---------------------------------------------------------------------------
# Markdown rendering for prompt injection
# ---------------------------------------------------------------------------

# Stamping fields the C generator stamps programmatically — strip from
# few-shot to avoid teaching the LLM to emit them.
_STAMPED_KEYS = ("session_id", "session_type", "persona_id", "rule_id", "rule_ref")

# Labels are also stripped at render time — under label-programmatisation
# the gen LLM must NOT emit labels. We surface `gold_markers` instead
# (reverse-derived from session.required_actions) so the few-shot teaches
# the new schema.
_RENDER_STRIP_KEYS = _STAMPED_KEYS + ("labels", "domain", "hidden_rule_slots", "gold_markers")

# Rule fields shown to the LLM in the exemplar header (tier-1 + tier-2).
_RULE_FIELDS = (
    "rule_id", "rule_text", "canonical_answer",
    "counterfactual_default",
    "check_type", "action_step",
)


def _strip_rule(rule: dict) -> dict:
    return {k: rule[k] for k in _RULE_FIELDS if k in rule}


def _render_session_for_example(rule: dict, session: dict) -> dict:
    """Return the session form shown to the LLM in few-shots.

    Trimmed to what actually teaches the new 3-field LLM output schema:
      instruction / gold_value / references.

    Dropped at render time:
      - Stamped identity fields (LLM doesn't emit them — caller stamps).
      - `labels` (caller derives from rule + gold_value).
      - `local_env.tools` (caller overwrites with whole-domain list).
      - `domain` (caller derives from rule.action_step).
      - `hidden_rule_slots` (not in schema).
      - `gold_markers` (we render `gold_value` instead).

    The decoy design signal lives in references.attributes; that's the
    teaching core.
    """
    refs = (session.get("local_env") or {}).get("references") or []
    out: dict = {
        "instruction": session.get("instruction"),
        "gold_value": _derive_gold_value(rule, session),
        "references": refs,
    }
    return out


def render_exemplars_md(exemplars: list[dict]) -> str:
    """Render a list of exemplars as markdown for prompt injection.

    Empty input → "(no exemplars)" so the placeholder doesn't read awkwardly.
    """
    if not exemplars:
        return "(no exemplars available)"

    blocks: list[str] = []
    for i, ex in enumerate(exemplars, 1):
        rule_compact = _strip_rule(ex["rule"])
        session_compact = _render_session_for_example(ex["rule"], ex["session"])
        blocks.append(
            f"### Example {i} — bucket=`{ex['bucket']}`, domain=`{ex['domain']}`\n"
            f"\n"
            f"**Rule**\n"
            f"```json\n"
            f"{json.dumps(rule_compact, ensure_ascii=False, indent=2)}\n"
            f"```\n"
            f"\n"
            f"**Test session** (caller stamps identity fields + derives "
            f"`labels` from rule + `gold_value`)\n"
            f"```json\n"
            f"{json.dumps(session_compact, ensure_ascii=False, indent=2)}\n"
            f"```"
        )
    return "\n\n".join(blocks)
