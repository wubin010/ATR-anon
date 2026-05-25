"""Pool audit: per-rule signal coverage + pure-noise classification.

Reads a persona's `test_sessions/` (truth source for coverage tools) and
`learning_sessions/` (the learning pool), and produces an `audit.json`
that drives downstream `compose.py` sampling.

Per-rule fields:
  rule_id      — id from test_sessions
  tools        — set of decision-surface tools covered by the rule's final
                 TS gold (from labels.task_success.required_actions). For
                 confirm rules, this is target_tool, not the generic
                 get_user_confirmation wrapper.
  N_natural    — count of LS in the pool whose decision-surface tools
                 overlap the rule's TS-derived coverage tools
  signal_ls    — list of those LS ids, sorted by session_id

Pool-level fields:
  pure_noise_ls — LS whose decision-surface tools do NOT overlap any rule's
                  TS-derived coverage tools. These are the "noise pool" —
                  episode-invariant.

CLI:
    uv run python -m datagen.episodes.audit --persona-id <pid>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datagen._common import PERSONAS_DIR, read_json, write_json

logger = logging.getLogger(__name__)


def _coverage_tools_from_steps(steps: list[dict]) -> set[str]:
    """Return decision-surface tools from TS RequiredAction / LS GoldStep data.

    `get_user_confirmation` is a generic wrapper, so it is not a useful
    coverage surface by itself. For confirm calls, use `arguments.target_tool`
    instead. This keeps "confirm before booking" from matching unrelated
    "confirm before deleting" learning sessions.
    """
    out: set[str] = set()
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if tool == "get_user_confirmation":
            args = step.get("arguments") or {}
            if isinstance(args, dict):
                target_tool = args.get("target_tool")
                if target_tool:
                    out.add(target_tool)
            continue
        if tool:
            out.add(tool)
    return out


def _required_actions(ts: dict) -> list[dict]:
    labels = ts.get("labels") or {}
    task_success = labels.get("task_success") or {}
    required = task_success.get("required_actions") or []
    return required if isinstance(required, list) else []


def _load_rule_tools(pdir: Path) -> dict[str, set[str]]:
    """rule_id → set of TS-derived coverage tools.

    Only rules with a corresponding test_session are included (downstream
    audit/compose only cares about evaluable rules). `rules.json` is only used
    to ensure the rule metadata exists for compose; coverage tools come from
    `test_session.labels.task_success.required_actions`, not the authoring
    artifact `rules.json.action_step`.
    """
    out: dict[str, set[str]] = {}
    ts_dir = pdir / "test_sessions"
    rules_path = pdir / "rules.json"
    if not ts_dir.exists() or not rules_path.exists():
        return out
    rule_ids = {r["rule_id"] for r in read_json(rules_path)}
    for f in sorted(ts_dir.glob("*.json")):
        t = read_json(f)
        rid = t.get("rule_id")
        if not rid or rid not in rule_ids:
            continue
        out[rid] = _coverage_tools_from_steps(_required_actions(t))
    return out


def _load_ls_tools(pdir: Path) -> dict[str, dict]:
    """ls_id → {tools, day_offset, domain}.

    Prefer decision-surface tools reverse-derived from `gold_trajectory`.
    Minimal LS files may have no trajectory, so fall back to
    `expected_tools` while still excluding the generic confirmation wrapper.
    """
    out: dict[str, dict] = {}
    ls_dir = pdir / "learning_sessions"
    if not ls_dir.exists():
        return out
    for f in sorted(ls_dir.glob("*.json")):
        s = read_json(f)
        expected_tools = set(s.get("expected_tools") or [])
        tools = _coverage_tools_from_steps(s.get("gold_trajectory") or [])
        if not tools:
            tools = set(expected_tools)
            tools.discard("get_user_confirmation")
        out[s["session_id"]] = {
            "tools": tools,
            "day_offset": s.get("day_offset", 0),
            "domain": s.get("domain", ""),
        }
    return out


def _single_tool(tools: set[str]) -> str | None:
    """Backward-compatible singular decision_tool field for one-tool rules."""
    if len(tools) != 1:
        return None
    return next(iter(tools))


def compute_audit(persona_id: str) -> dict:
    pdir = PERSONAS_DIR / persona_id
    rule_tools = _load_rule_tools(pdir)
    ls_meta = _load_ls_tools(pdir)

    rules_out = []
    for rid in sorted(rule_tools):
        rt = rule_tools[rid]
        signal_ls = sorted(sid for sid, m in ls_meta.items() if rt & m["tools"])
        # Diagnostic: which signal_ls also carry the rule's decision tool
        # (kept as a stable field for existing audit readers). Since `tools`
        # is now already TS-derived decision-surface coverage, this normally
        # equals `signal_ls`.
        dt = _single_tool(rt)
        decision_signal_ls = sorted(
            sid for sid in signal_ls
            if dt and dt in ls_meta[sid]["tools"]
        )
        rules_out.append({
            "rule_id": rid,
            "tools": sorted(rt),
            "decision_tool": dt,
            "decision_tools": sorted(rt),
            "N_natural": len(signal_ls),
            "N_decision": len(decision_signal_ls),
            "signal_ls": signal_ls,
            "decision_signal_ls": decision_signal_ls,
        })

    pure_noise = sorted(
        sid for sid, m in ls_meta.items()
        if all(not (rt & m["tools"]) for rt in rule_tools.values())
    )

    signal_pool = sorted(set().union(*[set(r["signal_ls"]) for r in rules_out])) if rules_out else []
    unteachable = [r["rule_id"] for r in rules_out if r["N_natural"] == 0]
    decision_only_signal_pool = sorted(set().union(*[set(r["decision_signal_ls"]) for r in rules_out])) if rules_out else []

    audit = {
        "persona_id": persona_id,
        "rules": rules_out,
        "pure_noise_ls": pure_noise,
        "unteachable_rules": unteachable,
        "summary": {
            "total_rules": len(rule_tools),
            "total_ls": len(ls_meta),
            "signal_pool_size_dedup": len(signal_pool),
            "decision_signal_pool_size_dedup": len(decision_only_signal_pool),
            "noise_pool_size": len(pure_noise),
            "rules_with_zero_signal": unteachable,
            "rules_with_zero_decision_signal": [
                r["rule_id"] for r in rules_out if r["N_decision"] == 0
            ],
        },
    }
    return audit


def run_persona(persona_id: str, out_path: Path | None = None) -> Path:
    audit = compute_audit(persona_id)
    if out_path is None:
        out_path = PERSONAS_DIR / persona_id / "audit.json"
    write_json(out_path, audit)
    s = audit["summary"]
    logger.info(
        "[%s] wrote %s — rules=%d (zero-signal=%d) ls=%d signal_pool=%d noise_pool=%d",
        persona_id, out_path, s["total_rules"], len(s["rules_with_zero_signal"]),
        s["total_ls"], s["signal_pool_size_dedup"], s["noise_pool_size"],
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pool audit for one persona")
    parser.add_argument("--persona-id", type=str, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_persona(args.persona_id, args.out)


if __name__ == "__main__":
    main()
