"""Unified datagen pipeline entry — orchestrates ingest → the rule stage →
the test-session stage → the learning-session stage → episode compose.

Designed for per-persona serial runs (failures stay isolated, log is readable):

  - ingest runs once in batch (parquet load is shared cost)
  - Then per persona: rules → test_session_gen → skeleton → fill → audit → compose

Each downstream stage is invoked via `subprocess.run` calling the existing
stage CLI — no internal imports, so a stage crash doesn't poison the
parent process and hot-fixes to a single stage don't require restarting
the orchestrator.

Usage examples:

  # Default formal 20-persona cohort, full pipeline (one episode per persona)
  uv run python -m datagen.pipeline

  # Custom persona selection with rule count + refine budget
  uv run python -m datagen.pipeline --selection data/persona_selection_formal20.json \
    --layer 40 --rule-count 25 --max-refine 3

  # Skip ingest (raw/structured already present), only re-run downstream
  uv run python -m datagen.pipeline --skip-a0

  # Single persona by slug or uuid prefix
  uv run python -m datagen.pipeline --persona-id charlie_james

  # Compose alone (audit + compose only — assumes all upstream done)
  uv run python -m datagen.pipeline --only compose
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from datagen._common import PERSONAS_DIR, PROJECT_ROOT
from datagen.config import CONFIG
from datagen.ingest.ingest import (
    DEFAULT_SELECTION,
    load_uuids_from_selection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("datagen.run")


# Stage labels (also CLI tags for --only / --skip)
STAGE_A0 = "a0"
STAGE_RULES = "rules"
STAGE_TEST = "test"
STAGE_SKELETON = "skeleton"
STAGE_FILL = "fill"
STAGE_AUDIT = "audit"
STAGE_COMPOSE = "compose"

# Downstream stages run per-persona in this order.
PERSONA_STAGES = [
    STAGE_RULES, STAGE_TEST, STAGE_SKELETON, STAGE_FILL, STAGE_AUDIT, STAGE_COMPOSE,
]
ALL_STAGES = [STAGE_A0] + PERSONA_STAGES


@dataclass
class StageResult:
    persona_id: str
    stage: str
    status: str          # "ok" | "fail" | "skip"
    duration_s: float
    metrics: dict = field(default_factory=dict)
    err_tail: str = ""   # last few lines of stderr if fail


def _run_subprocess(cmd: list[str], log_prefix: str) -> tuple[int, str]:
    """Run a stage CLI, stream its stdout/stderr to our logger,
    and return (returncode, last_8_lines_combined)."""
    logger.info("%s exec: %s", log_prefix, " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT, text=True, bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  | {line}", flush=True)
            tail.append(line)
            if len(tail) > 30:
                tail = tail[-30:]
    proc.wait()
    return proc.returncode, "\n".join(tail[-8:])


def _persona_metrics(pid: str) -> dict:
    """Snapshot per-persona output counts for the post-stage summary."""
    pdir = PERSONAS_DIR / pid
    out: dict = {}

    rules_p = pdir / "rules.json"
    if rules_p.exists():
        try:
            out["rules"] = len(json.loads(rules_p.read_text()))
        except Exception:
            out["rules"] = "?"

    ts_dir = pdir / "test_sessions"
    out["ts"] = len(list(ts_dir.iterdir())) if ts_dir.is_dir() else 0

    ls_dir = pdir / "learning_sessions"
    out["ls"] = len(list(ls_dir.iterdir())) if ls_dir.is_dir() else 0

    eps_dir = pdir / "episodes"
    out["episodes"] = len(list(eps_dir.iterdir())) if eps_dir.is_dir() else 0

    return out


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def stage_a0(args, target_uuids: set[str]) -> list[StageResult]:
    """Batch ingest. Returns one StageResult per persona created."""
    cmd = [
        "uv", "run", "python", "-m", "datagen.ingest.ingest",
        "--selection", str(args.selection),
    ]
    if args.layer is not None:
        cmd += ["--layer", str(args.layer)]
    if args.force:
        cmd += ["--force"]

    t0 = time.time()
    rc, tail = _run_subprocess(cmd, "[ingest]")
    dur = time.time() - t0
    status = "ok" if rc == 0 else "fail"

    # Map back from selection uuids → resulting slug dirs
    results: list[StageResult] = []
    for u in sorted(target_uuids):
        # find slug dir whose raw.original_uuid == u
        slug = _find_slug_by_uuid(u)
        results.append(StageResult(
            persona_id=slug or f"<uuid:{u[:12]}>",
            stage=STAGE_A0, status=status, duration_s=dur,
            metrics={"resolved_slug": bool(slug)},
            err_tail=tail if status == "fail" else "",
        ))
    return results


def _find_slug_by_uuid(original_uuid: str) -> str | None:
    if not PERSONAS_DIR.is_dir():
        return None
    for d in PERSONAS_DIR.iterdir():
        if not d.is_dir():
            continue
        raw_p = d / "raw.json"
        if not raw_p.exists():
            continue
        try:
            data = json.loads(raw_p.read_text())
        except Exception:
            continue
        if (data.get("nemotron_uuid") or data.get("original_uuid")) == original_uuid:
            return d.name
    return None


def stage_rules(args, pid: str) -> StageResult:
    cmd = ["uv", "run", "python", "-m", "datagen.rules.pipeline",
           "--persona-id", pid,
           "--n", str(args.rule_count)]
    if args.force:
        cmd += ["--force"]
    t0 = time.time()
    rc, tail = _run_subprocess(cmd, f"[A/rules {pid}]")
    return StageResult(pid, STAGE_RULES,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


def stage_test(args, pid: str) -> StageResult:
    cmd = ["uv", "run", "python", "-m", "datagen.test_sessions.pipeline",
           "--persona-id", pid,
           "--max-refine", str(args.max_refine)]
    if args.force:
        cmd += ["--force"]
    t0 = time.time()
    rc, tail = _run_subprocess(cmd, f"[C/test {pid}]")
    return StageResult(pid, STAGE_TEST,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


def stage_skeleton(args, pid: str) -> StageResult:
    cmd = ["uv", "run", "python", "-m", "datagen.learning_sessions.skeleton.gen",
           "--persona-id", pid]
    if args.force:
        cmd += ["--force"]
    t0 = time.time()
    rc, tail = _run_subprocess(cmd, f"[skeleton {pid}]")
    return StageResult(pid, STAGE_SKELETON,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


def stage_fill(args, pid: str) -> StageResult:
    cmd = ["uv", "run", "python", "-m", "datagen.learning_sessions.fill.gen",
           "--persona-id", pid]
    if args.force:
        cmd += ["--force"]
    t0 = time.time()
    rc, tail = _run_subprocess(cmd, f"[fill {pid}]")
    return StageResult(pid, STAGE_FILL,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


def stage_audit(args, pid: str) -> StageResult:
    cmd = ["uv", "run", "python", "-m", "datagen.episodes.audit",
           "--persona-id", pid]
    t0 = time.time()
    rc, tail = _run_subprocess(cmd, f"[D/audit {pid}]")
    return StageResult(pid, STAGE_AUDIT,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


def stage_compose(args, pid: str) -> StageResult:
    """Compose the persona's episode (one per persona)."""
    t0 = time.time()
    cmd = ["uv", "run", "python", "-m", "datagen.episodes.compose",
           "--persona-id", pid]
    rc, tail = _run_subprocess(cmd, f"[D/compose {pid}]")
    return StageResult(pid, STAGE_COMPOSE,
                       "ok" if rc == 0 else "fail",
                       time.time() - t0, _persona_metrics(pid),
                       err_tail=tail if rc != 0 else "")


PERSONA_STAGE_FNS = {
    STAGE_RULES:    stage_rules,
    STAGE_TEST:     stage_test,
    STAGE_SKELETON: stage_skeleton,
    STAGE_FILL:     stage_fill,
    STAGE_AUDIT:    stage_audit,
    STAGE_COMPOSE:  stage_compose,
}


# ---------------------------------------------------------------------------
# Persona discovery
# ---------------------------------------------------------------------------


def _discover_personas(args) -> tuple[set[str], list[str]]:
    """Returns (target_uuids, known_persona_ids).

    target_uuids = uuids from selection.json (for ingest).
    known_persona_ids = existing persona dirs that match the filter
                       (for downstream stages that need a slug — ingest may
                       create new dirs, those get added after ingest runs).
    """
    target_uuids = load_uuids_from_selection(args.selection, layer=args.layer)
    if args.persona_id:
        # Filter by uuid prefix OR existing slug prefix
        keep_uuids: set[str] = set()
        for u in target_uuids:
            if u.startswith(args.persona_id):
                keep_uuids.add(u)
                continue
            slug = _find_slug_by_uuid(u)
            if slug and slug.startswith(args.persona_id):
                keep_uuids.add(u)
        target_uuids = keep_uuids

    # Known persona dirs already on disk that map to selection uuids
    known: list[str] = []
    for u in sorted(target_uuids):
        slug = _find_slug_by_uuid(u)
        if slug:
            known.append(slug)
    return target_uuids, known


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results: list[StageResult]) -> None:
    if not results:
        return
    # Group by persona, latest stage wins for metrics snapshot
    by_pid: dict[str, dict] = {}
    for r in results:
        slot = by_pid.setdefault(r.persona_id, {"stages": [], "metrics": {}})
        slot["stages"].append((r.stage, r.status, r.duration_s))
        if r.metrics:
            slot["metrics"] = r.metrics

    print()
    print("=" * 92)
    print(f"{'persona':<25} {'rules':>6} {'ts':>4} {'ls':>4} {'eps':>4}  stages (status)")
    print("-" * 92)
    fail_total = 0
    for pid in sorted(by_pid):
        d = by_pid[pid]
        m = d["metrics"]
        stages_str = " → ".join(f"{s}={st}" for s, st, _ in d["stages"])
        if any(st == "fail" for _, st, _ in d["stages"]):
            fail_total += 1
        print(f"{pid:<25} {m.get('rules','?'):>6} {m.get('ts',0):>4} "
              f"{m.get('ls',0):>4} {m.get('episodes',0):>4}  {stages_str}")
    print("=" * 92)
    total_dur = sum(r.duration_s for r in results)
    print(f"total wall time: {total_dur/60:.1f} min  ({len(by_pid)} personas, "
          f"{len(results)} stage runs, {fail_total} persona(s) with failures)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="datagen unified pipeline runner")
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                   help="persona selection JSON (default: data/persona_selection_formal20.json)")
    p.add_argument("--layer", type=int, default=None,
                   choices=[5, 10, 20, 40, 100],
                   help="selection layer subset (default: all entries in selection)")
    p.add_argument("--persona-id", type=str, default=None,
                   help="prefix-match against uuid[:12] OR existing dir slug")
    p.add_argument("--rule-count", type=int, default=CONFIG.rule_count_per_persona,
                   help=f"rule-stage rules N (default {CONFIG.rule_count_per_persona} from datagen.config)")
    p.add_argument("--max-refine", type=int, default=CONFIG.test_max_refine_rounds,
                   help=f"test-session refine rounds (default {CONFIG.test_max_refine_rounds} from datagen.config)")
    p.add_argument("--skip-a0", action="store_true",
                   help="skip ingest (raw/structured already on disk)")
    p.add_argument("--skip-stages", type=str, default="",
                   help="comma-separated stages to skip "
                        f"(any of {','.join(ALL_STAGES)})")
    p.add_argument("--only", type=str, default=None,
                   help="run ONLY these comma-separated stages "
                        f"(any of {','.join(ALL_STAGES)})")
    p.add_argument("--force", action="store_true", default=True,
                   help="pass --force to all stages (default True)")
    p.add_argument("--no-force", dest="force", action="store_false",
                   help="disable --force (resume mode: skip already-done outputs)")
    args = p.parse_args()

    # Resolve which stages to run
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = only_set - set(ALL_STAGES)
        if unknown:
            p.error(f"--only: unknown stage(s) {sorted(unknown)}")
        stages_to_run = [s for s in ALL_STAGES if s in only_set]
    else:
        skip_set = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
        unknown = skip_set - set(ALL_STAGES)
        if unknown:
            p.error(f"--skip-stages: unknown stage(s) {sorted(unknown)}")
        if args.skip_a0:
            skip_set.add(STAGE_A0)
        stages_to_run = [s for s in ALL_STAGES if s not in skip_set]

    logger.info("stages: %s", " → ".join(stages_to_run))
    logger.info("force=%s, rule_count=%d, max_refine=%d",
                args.force, args.rule_count, args.max_refine)

    # Discover targets
    target_uuids, known_pids = _discover_personas(args)
    if not target_uuids:
        logger.error("no personas matched selection filter; nothing to do")
        sys.exit(1)
    logger.info("targets: %d uuid(s)%s",
                len(target_uuids),
                f" (matching --persona-id {args.persona_id!r})" if args.persona_id else "")

    results: list[StageResult] = []

    # ingest (batch)
    if STAGE_A0 in stages_to_run:
        logger.info("=" * 60)
        logger.info("=== ingest (batch) ===")
        logger.info("=" * 60)
        results.extend(stage_a0(args, target_uuids))
        # Re-discover to pick up freshly-created slug dirs
        _, known_pids = _discover_personas(args)

    if not known_pids:
        logger.error("no persona dirs resolved after ingest; cannot run downstream")
        _print_summary(results)
        sys.exit(1)

    # Persona stages — serial per persona, all stages per persona before next
    persona_stages = [s for s in stages_to_run if s in PERSONA_STAGES]
    for pid in known_pids:
        logger.info("=" * 60)
        logger.info("=== persona: %s ===", pid)
        logger.info("=" * 60)
        for stage in persona_stages:
            r = PERSONA_STAGE_FNS[stage](args, pid)
            results.append(r)
            if r.status == "fail":
                logger.error("[%s] stage %s FAILED, last lines:\n%s",
                             pid, stage, r.err_tail)
                logger.warning("[%s] aborting remaining stages for this persona",
                               pid)
                break

    _print_summary(results)
    fail_count = sum(1 for r in results if r.status == "fail")
    sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()
