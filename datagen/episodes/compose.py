"""Compose one episode from a persona's rule + session pool.

Each persona yields exactly one episode per seed.

Sampling spec:
  - test_sessions / rules: full set (all rules that passed C QC, including
    zero-signal ones — valid test items where the agent can only guess).
  - K (trajectory length) = total_test_sessions * 2.
  - learning_sessions: every teachable rule (signal LS > 0) contributes its
    signal LS; pure-noise LS then fill the remainder up to K (no
    replacement — if the noise pool is short, the trajectory ends up
    shorter than K).
  - Order: ascending by `day_offset` (then session_id).

Output: data/personas/<pid>/episodes/<pid>_seed{YYY}.json

CLI:
    uv run python -m datagen.episodes.compose --persona-id <pid> --seed 0 [--K <int>]
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from datagen._common import PERSONAS_DIR, read_json, write_json
from datagen.episodes.audit import compute_audit

logger = logging.getLogger(__name__)


def _strip_rule_private(rule: dict) -> dict:
    """Drop rule QC metadata + authoring artifact fields for clean Episode dump."""
    out = dict(rule)
    out.pop("_qc", None)
    out.pop("_reject_reasons", None)
    out.pop("persona_id", None)
    return out


def _sample_noise(noise_pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Sample k noise ids; capped at pool size (no replacement)."""
    if k <= 0:
        return []
    return rng.sample(noise_pool, min(k, len(noise_pool)))


def compose_episode(
    persona_id: str,
    seed: int,
    K: int | None = None,
    out_path: Path | None = None,
) -> Path:
    pdir = PERSONAS_DIR / persona_id
    audit = compute_audit(persona_id)
    if not audit["rules"]:
        raise RuntimeError(f"[{persona_id}] no rules in test_sessions/, cannot compose")

    rng = random.Random(seed)

    # All rules with test sessions — full test bench (including zero-signal).
    rule_ids = [r["rule_id"] for r in audit["rules"]]
    audit_rule_signal = {r["rule_id"]: set(r["signal_ls"]) for r in audit["rules"]}

    # Every teachable rule (signal LS > 0) contributes its signal to the
    # learning phase; zero-signal rules stay in the test bench but supply
    # no learning evidence (the agent can only guess on them).
    teachable_ids = [rid for rid in rule_ids if audit_rule_signal[rid]]
    selected_rules = sorted(teachable_ids)

    # K = total test sessions × 2
    if K is None:
        K = len(rule_ids) * 2

    # ── signal LS (dedup across teachable rules) ─────────────────────────
    signal_set = (
        set().union(*[audit_rule_signal[rid] for rid in selected_rules])
        if selected_rules else set()
    )

    # If signal exceeds K, truncate; otherwise fill the gap with pure noise.
    if len(signal_set) > K:
        signal_ls = sorted(rng.sample(sorted(signal_set), K))
        noise_ls: list[str] = []
    else:
        gap = K - len(signal_set)
        noise_ls = _sample_noise(audit["pure_noise_ls"], gap, rng)
        signal_ls = sorted(signal_set)

    actual_signal_overlap_rules = sorted(
        rid for rid in rule_ids if audit_rule_signal[rid] & set(signal_ls)
    )
    selected_rule_fraction = len(selected_rules) / len(rule_ids)

    # ── load full session objects ────────────────────────────────────────
    ls_obj = {read_json(f)["session_id"]: read_json(f) for f in (pdir / "learning_sessions").glob("*.json")}
    selected_session_ids = signal_ls + noise_ls
    selected_sessions = [ls_obj[sid] for sid in selected_session_ids]
    # natural daily order (session_id used as deterministic tiebreaker)
    selected_sessions.sort(key=lambda s: (s.get("day_offset", 0), s.get("session_id", "")))

    # ── test_sessions / rules: full set ──────────────────────────────────
    ts_obj = {read_json(f)["rule_id"]: read_json(f) for f in (pdir / "test_sessions").glob("*.json")}
    kept_test = [ts_obj[rid] for rid in rule_ids if rid in ts_obj]

    rules_all = read_json(pdir / "rules.json")
    rules_by_id = {r["rule_id"]: r for r in rules_all}
    kept_rules = [_strip_rule_private(rules_by_id[rid]) for rid in rule_ids if rid in rules_by_id]

    raw = read_json(pdir / "raw.json")
    structured = read_json(pdir / "structured.json")

    episode = {
        "episode_id": f"{persona_id}_seed{seed:03d}",
        "persona_id": persona_id,
        "raw_persona": raw,
        "structured_persona": structured,
        "rules": kept_rules,
        "learning_sessions": selected_sessions,
        "test_sessions": kept_test,
        "metadata": {
            "selected_rule_fraction": selected_rule_fraction,
            "actual_signal_overlap_rules": actual_signal_overlap_rules,
            "actual_signal_overlap_rule_fraction": (
                len(actual_signal_overlap_rules) / len(rule_ids)
            ),
            "teachable_rule_count": len(teachable_ids),
            "zero_signal_rule_count": len(rule_ids) - len(teachable_ids),
            "signal_pool_size_dedup": audit["summary"]["signal_pool_size_dedup"],
            "noise_pool_size": audit["summary"]["noise_pool_size"],
            "K": K,
            "seed": seed,
            "selected_rules": selected_rules,
            "extra_signal_rules": [],
            "all_test_rules": rule_ids,
            "signal_count": len(signal_ls),
            "noise_count": len(noise_ls),
            "trajectory_length": len(selected_session_ids),
        },
    }

    if out_path is None:
        out_dir = pdir / "episodes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{episode['episode_id']}.json"
    write_json(out_path, episode)
    logger.info(
        "[%s] wrote %s — rules=%d signal=%d noise=%d",
        persona_id, out_path, len(selected_rules), len(signal_ls), len(noise_ls),
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compose one episode for a persona")
    parser.add_argument("--persona-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=None,
                        help="trajectory length (default: test_count × 2 per persona)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    compose_episode(args.persona_id, args.seed, args.K, args.out)


if __name__ == "__main__":
    main()
