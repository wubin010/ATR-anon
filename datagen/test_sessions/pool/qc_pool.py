"""Run QC on the test_session_pool — no generation, no refine.

Pool QC concurrency model (different from gen pipeline's qc_test):
  - **Sessions are processed serially** (one at a time). Avoids slot-thrashing
    across the per-model semaphore when 60+ sessions × 3 ensemble samples
    all queue up FIFO and individual sessions wait forever for their last
    sample to complete.
  - **Within each session, all 6 ensemble calls (3 upper + 3 lower) fire
    in parallel** — single small ThreadPool of 6 workers. With concurrency
    cap 20 in lib.llm, all 6 of one session's calls land slots immediately,
    so wall-clock per session ≈ slowest single call.
  - **No lazy-eval** (always run lower even when upper fails). For pool
    diagnostics we want both numbers to inspect when something fails.

Reads `data/test_session_pool/index.json` + per-session JSON files +
rules.json, writes `qc_pool_results.json` with upper/lower per-sample
detail.

Used to validate pool quality before wiring into gen/refine few-shot.

CLI:
    uv run python -m datagen.test_sessions.pool.qc_pool \
        [--limit N] [--rule-id-prefix <pfx>] [--qc-level full|oracle] [--qc-n 3]
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datagen._common import PROJECT_ROOT, read_json, write_json
from datagen.test_sessions.qc import (
    DEFAULT_QC_MODELS,
    QcLevel,
    QcReport,
    _representative_index,
    _upper_failure_msg,
    _lower_failure_msg,
    _extract_reason,
    _auto_upgrade_level,
    llm_qc_trace,
    trace_matches_gold,
)
from datagen.test_sessions.static_check import static_check

POOL_DIR = PROJECT_ROOT / "data" / "few_shot_pool" / "test_session_gen"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _qc_session_parallel_bounds(
    session: dict,
    rule: dict,
    qc_level: QcLevel,
    qc_models: tuple[str, ...],
) -> QcReport:
    """Pool-specific QC: dispatch all upper + lower ensemble calls together.

    Returns a QcReport populated like the gen-side `qc_test`, but obtained
    by firing 2 × len(qc_models) calls in a single ThreadPool batch
    (no lazy-eval, no probe). Static check still runs first as a cheap gate.
    """
    effective_level = _auto_upgrade_level(qc_level, rule)
    static_errors = static_check(session, rule)

    report = QcReport(
        passed=False,
        level=effective_level,
        n_samples=len(qc_models),
        qc_models=list(qc_models),
        static_errors=static_errors,
    )

    if static_errors:
        report.failure_reasons = list(static_errors)
        return report

    if effective_level == QcLevel.STATIC:
        report.passed = True
        return report

    n_total = len(qc_models)
    if n_total == 0:
        raise ValueError("qc_models must be non-empty")

    # Build the 6 (or 2N) tasks: N upper (rule_injected=True) + N lower (False).
    # Order in batch_results matches `tasks` order so we can split back cleanly.
    sid = session.get("session_id", "?")
    upper_tasks = [(True, m, i) for i, m in enumerate(qc_models)]
    lower_tasks = [(False, m, i) for i, m in enumerate(qc_models)]
    full_qc = effective_level == QcLevel.FULL
    tasks = upper_tasks + lower_tasks if full_qc else upper_tasks

    def _one(t):
        rule_injected, model, idx = t
        bound = "upper" if rule_injected else "lower"
        tag = f"[{sid}] {bound}#{idx} model={model}"
        logger.info("%s start", tag)
        t0 = time.time()
        try:
            trace, prompt, raw, err = llm_qc_trace(session, rule_injected, rule, model)
        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("%s elapsed=%.1fs EXC=%s", tag, elapsed, e)
            raise
        elapsed = time.time() - t0
        if err:
            logger.warning("%s elapsed=%.1fs err=%s", tag, elapsed, err)
        else:
            n_steps = len(trace) if isinstance(trace, list) else 0
            logger.info("%s elapsed=%.1fs ok trace_steps=%d", tag, elapsed, n_steps)
        return trace, prompt, raw, err

    # Single small pool — all calls fire concurrently. lib.llm's per-model
    # semaphore (cap 20) admits them immediately when running serially.
    logger.info("[%s] dispatching %d calls (%d upper + %d lower) in parallel",
                sid, len(tasks), len(upper_tasks),
                len(lower_tasks) if full_qc else 0)
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        results = list(ex.map(_one, tasks))

    upper_results = results[:n_total]
    lower_results = results[n_total:] if full_qc else []

    gold = session["labels"]["task_success"]["required_actions"]

    # ── Upper aggregation ──
    upper_traces, upper_prompts, upper_raws, upper_errors = zip(*upper_results)
    upper_traces = list(upper_traces)
    upper_prompts = list(upper_prompts)
    upper_raws = list(upper_raws)
    upper_errors = list(upper_errors)
    upper_reasons = [_extract_reason(r) for r in upper_raws]
    upper_matches = [trace_matches_gold(t, gold) for t in upper_traces]
    upper_ok_idx = [i for i, e in enumerate(upper_errors) if e is None]
    upper_n_ok = len(upper_ok_idx)
    upper_hits = sum(upper_matches[i] for i in upper_ok_idx)
    upper_matched = upper_n_ok > 0 and upper_hits * 3 >= upper_n_ok * 2

    rep_u = _representative_index(upper_matches, upper_errors, prefer_match=True)
    report.upper_traces = upper_traces
    report.upper_reasons = upper_reasons
    report.upper_errors = upper_errors
    report.upper_match_per_sample = upper_matches
    report.upper_hits = upper_hits
    report.upper_matched = upper_matched
    report.upper_trace = upper_traces[rep_u]
    report.upper_reason = upper_reasons[rep_u]
    report.upper_prompt = upper_prompts[rep_u]
    report.upper_raw_response = upper_raws[rep_u]

    if upper_n_ok == 0:
        distinct = sorted({e for e in upper_errors if e})
        report.llm_infra_failed = True
        report.failure_reasons.append(
            f"llm_infra_error_all_upper_samples: {distinct}"
        )
        # We may still have lower data; populate it but the verdict is infra.
        if lower_results:
            _populate_lower(report, lower_results, gold, n_total)
        return report

    if effective_level == QcLevel.ORACLE:
        if upper_matched:
            report.passed = True
        else:
            report.failure_reasons.append(
                _upper_failure_msg(upper_reasons, upper_errors,
                                   upper_hits, upper_n_ok, n_total)
            )
        return report

    # FULL
    _populate_lower(report, lower_results, gold, n_total)

    if not upper_matched:
        report.failure_reasons.append(
            _upper_failure_msg(upper_reasons, upper_errors,
                               upper_hits, upper_n_ok, n_total)
        )
    if report.lower_matched:
        report.failure_reasons.append(
            _lower_failure_msg(report.lower_reasons or [],
                               report.lower_match_per_sample or [],
                               report.lower_errors or [],
                               report.lower_hits or 0,
                               sum(1 for e in (report.lower_errors or []) if e is None),
                               n_total)
        )
    report.passed = upper_matched and not report.lower_matched
    return report


def _populate_lower(
    report: QcReport,
    lower_results: list[tuple],
    gold: list[dict],
    n_total: int,
) -> None:
    if not lower_results:
        return
    lower_traces, lower_prompts, lower_raws, lower_errors = zip(*lower_results)
    lower_traces = list(lower_traces)
    lower_prompts = list(lower_prompts)
    lower_raws = list(lower_raws)
    lower_errors = list(lower_errors)
    lower_reasons = [_extract_reason(r) for r in lower_raws]
    lower_matches = [trace_matches_gold(t, gold) for t in lower_traces]
    lower_ok_idx = [i for i, e in enumerate(lower_errors) if e is None]
    lower_n_ok = len(lower_ok_idx)
    lower_hits = sum(lower_matches[i] for i in lower_ok_idx)
    lower_matched = lower_n_ok > 0 and lower_hits * 3 >= lower_n_ok * 2

    rep_l = _representative_index(lower_matches, lower_errors, prefer_match=True)
    report.lower_traces = lower_traces
    report.lower_reasons = lower_reasons
    report.lower_errors = lower_errors
    report.lower_match_per_sample = lower_matches
    report.lower_hits = lower_hits
    report.lower_matched = lower_matched
    report.lower_trace = lower_traces[rep_l]
    report.lower_reason = lower_reasons[rep_l]
    report.lower_prompt = lower_prompts[rep_l]
    report.lower_raw_response = lower_raws[rep_l]

    if lower_n_ok == 0:
        distinct = sorted({e for e in lower_errors if e})
        report.llm_infra_failed = True
        report.failure_reasons.append(
            f"llm_infra_error_all_lower_samples: {distinct}"
        )


def qc_one(index_entry: dict, rules_by_id: dict[str, dict],
           qc_level: QcLevel, qc_models: tuple[str, ...]) -> dict:
    rid = index_entry["rule_id"]
    session_path = PROJECT_ROOT / index_entry["path"]
    session = read_json(session_path)
    rule = rules_by_id.get(rid)
    if rule is None:
        return {
            "rule_id": rid,
            "bucket": index_entry["bucket"],
            "error": "rule_not_found_in_rules_json",
        }

    t0 = time.time()
    report = _qc_session_parallel_bounds(session, rule, qc_level, qc_models)
    elapsed = time.time() - t0

    return {
        "rule_id": rid,
        "bucket": index_entry["bucket"],
        "domain": index_entry["domain"],
        "check_type": index_entry["check_type"],
        "passed": report.passed,
        "n_samples": report.n_samples,
        "qc_models": report.qc_models,
        "static_errors": report.static_errors,
        "upper_matched": report.upper_matched,
        "lower_matched": report.lower_matched,
        "upper_hits": report.upper_hits,
        "lower_hits": report.lower_hits,
        "upper_reason": report.upper_reason,
        "lower_reason": report.lower_reason,
        "upper_reasons": report.upper_reasons,
        "lower_reasons": report.lower_reasons,
        "upper_match_per_sample": report.upper_match_per_sample,
        "lower_match_per_sample": report.lower_match_per_sample,
        "upper_trace": report.upper_trace,
        "lower_trace": report.lower_trace,
        "failure_reasons": report.failure_reasons,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Run QC on test_session_pool")
    parser.add_argument("--limit", type=int, default=None,
                        help="QC at most N sessions")
    parser.add_argument("--rule-id-prefix", type=str, default=None,
                        help="filter rules whose id starts with this")
    parser.add_argument("--rule-ids", type=str, default=None,
                        help="comma-separated exact rule_ids (overrides --rule-id-prefix)")
    parser.add_argument("--qc-level", type=str, default="full",
                        choices=["static", "oracle", "full"])
    parser.add_argument("--qc-models", type=str,
                        default=",".join(DEFAULT_QC_MODELS),
                        help=("comma-separated list of QC samples (one LLM call per "
                              "entry per bound; repeats of same model = independent "
                              "samples, distinct models = ensemble). default "
                              f"{','.join(DEFAULT_QC_MODELS)}"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default data/test_session_pool/qc_pool_results.json)")
    parser.add_argument("--no-resume", action="store_true",
                        help=("by default, if --out file exists, already-PASSED entries "
                              "are skipped (FAILed entries get retried); pass --no-resume "
                              "to ignore the existing file and start fresh"))
    parser.add_argument("--session-workers", type=int, default=0,
                        help=("number of sessions QC'd concurrently (each session "
                              "fires 2N ensemble LLM calls in parallel internally). "
                              "default 0 = all sessions in parallel; total in-flight "
                              "LLM calls is bounded by lib.llm.PER_MODEL_MAX_CONCURRENCY"))
    args = parser.parse_args()

    qc_models = tuple(m.strip() for m in args.qc_models.split(",") if m.strip())

    index = read_json(POOL_DIR / "index.json")
    rules_list = read_json(POOL_DIR / "rules.json")
    rules_by_id = {r["rule_id"]: r for r in rules_list}

    if args.rule_ids:
        wanted = {s.strip() for s in args.rule_ids.split(",") if s.strip()}
        index = [e for e in index if e["rule_id"] in wanted]
    elif args.rule_id_prefix:
        index = [e for e in index if e["rule_id"].startswith(args.rule_id_prefix)]
    if args.limit:
        index = index[: args.limit]

    if not index:
        logger.info("nothing to QC")
        return

    qc_level = QcLevel(args.qc_level)

    # ── Resume support ──────────────────────────────────────────────────
    # Read prior results from out_path (if it exists). Already-PASSED
    # entries are kept and the corresponding sessions are skipped this run.
    # FAILed / errored entries are dropped — they'll be re-QC'd.
    out_path = args.out or (POOL_DIR / "qc_pool_results.json")
    prior_results: list[dict] = []
    skip_ids: set[str] = set()
    if out_path.exists() and not args.no_resume:
        try:
            prior_results = read_json(out_path)
            if not isinstance(prior_results, list):
                prior_results = []
        except Exception as e:
            logger.warning("--out exists but unreadable (%s) — starting fresh", e)
            prior_results = []
        # Keep only PASSED prior entries; FAILed get re-tried.
        keep = [r for r in prior_results if r.get("passed")]
        skip_ids = {r["rule_id"] for r in keep if r.get("rule_id")}
        prior_results = keep

    if skip_ids:
        before = len(index)
        index = [e for e in index if e["rule_id"] not in skip_ids]
        logger.info("resume: %d previously-PASSED entries kept; %d/%d sessions queued",
                    len(skip_ids), len(index), before)
        if not index:
            # Nothing left to QC — just preserve prior results.
            write_json(out_path, sorted(prior_results,
                                         key=lambda x: (x.get("bucket", ""), x.get("rule_id", ""))))
            logger.info("nothing to do — already complete")
            return

    # session-workers=0 (default) → all sessions in parallel; throttling
    # happens at lib.llm.PER_MODEL_MAX_CONCURRENCY (single semaphore caps
    # actual in-flight Gemini calls).
    workers = len(index) if args.session_workers <= 0 else min(args.session_workers, len(index))
    workers = max(1, workers)

    from lib.llm import PER_MODEL_MAX_CONCURRENCY  # type: ignore
    logger.info(
        "QC %d pool sessions, %d sessions concurrent (each fires %d × 2 "
        "ensemble calls in parallel) — actual in-flight Gemini calls capped "
        "by lib.llm.PER_MODEL_MAX_CONCURRENCY=%d, level=%s, qc_models=%s",
        len(index), workers, len(qc_models),
        PER_MODEL_MAX_CONCURRENCY,
        qc_level.value, list(qc_models),
    )

    # `results` accumulates resumed PASSes + this run's outcomes.
    # Lock-protected because multiple session workers may finish near-
    # simultaneously and append + snapshot the file. Snapshot writes are
    # atomic enough at the dict level under GIL but we still need to
    # serialize writers vs. concurrent appenders.
    results: list[dict] = list(prior_results)
    write_lock = threading.Lock()
    t0 = time.time()

    # Sessions are processed serially — one at a time. Within each session,
    # _qc_session_parallel_bounds fires upper × N + lower × N calls in
    # parallel and waits for all to return before moving to the next session.
    completed = [0]  # mutable counter shared across threads (lock-protected)

    def _process_one(entry: dict) -> dict:
        try:
            r = qc_one(entry, rules_by_id, qc_level, qc_models)
        except Exception as e:
            r = {
                "rule_id": entry["rule_id"],
                "bucket": entry["bucket"],
                "error": f"exception:{e}",
            }
        with write_lock:
            results.append(r)
            completed[0] += 1
            i = completed[0]
            if r.get("error"):
                logger.warning("[%d/%d] [%s] ERROR %s",
                               i, len(index), r["rule_id"], r["error"])
            else:
                status = "PASS" if r["passed"] else "FAIL"
                u_h = r.get("upper_hits")
                l_h = r.get("lower_hits")
                n = r.get("n_samples", 1)
                logger.info(
                    "[%d/%d] [%s] %-42s %s upper=%s/%s lower=%s/%s (%.1fs)",
                    i, len(index), r["bucket"], r["rule_id"], status,
                    u_h, n, l_h, n, r["elapsed_s"],
                )
            # Incremental write — persist after every session so a kill
            # mid-run leaves partial progress on disk. Sort for stable diffs.
            snapshot = sorted(
                results,
                key=lambda x: (x.get("bucket", ""), x.get("rule_id", "")),
            )
            write_json(out_path, snapshot)
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process_one, e) for e in index]
        # Drain in completion order — already logged inside _process_one.
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                logger.error("session worker crashed: %s", e, exc_info=True)

    results.sort(key=lambda x: (x.get("bucket", ""), x.get("rule_id", "")))

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    static_fail = sum(1 for r in results if r.get("static_errors"))
    upper_miss = sum(1 for r in results if r.get("upper_matched") is False)
    lower_hit = sum(1 for r in results if r.get("lower_matched") is True)
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("QC pool done in %.1fs", elapsed)
    logger.info("  total:        %d", total)
    logger.info("  passed:       %d", passed)
    logger.info("  static fail:  %d", static_fail)
    logger.info("  upper miss:   %d  (oracle agent did not reach gold)", upper_miss)
    logger.info("  lower hit:    %d  (passive agent accidentally hit gold)", lower_hit)
    logger.info("  results → %s", out_path)


if __name__ == "__main__":
    main()
