"""The test-session stage pipeline: per-rule gen → qc → refine loop, parallelized over rules.

For each rule:
    candidate = generate_candidate(...)
    for round in 0..MAX_REFINE:
        report = qc_test(candidate, rule, persona_brief, level=...)
        if report.passed: emit; break
        if round == MAX_REFINE: break
        candidate = refine_test(candidate, report, ...)
        candidate = stamp_session(candidate, ...)  # re-stamp after refine

Outputs:
    data/personas/<uuid>/test_sessions/<rule_id>.json   (accepted)
    data/personas/<uuid>/test_session_qc_log.json       (every attempt)
    data/personas/<uuid>/test_session_dropped.json      (rules with no accepted session)

CLI:
    uv run python -m datagen.test_sessions.pipeline \
        --persona-id <uuid> [--rule-id <rid>] [--qc-level full|oracle|static]
        [--max-refine 2] [--force]
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datagen._common import PERSONAS_DIR, read_json, write_json
from datagen.config import CONFIG
from datagen.test_sessions.gen import (
    CandidateGenerationInfraError,
    domain_of_rule,
    generate_candidate,
    stamp_session,
)
from datagen.test_sessions.qc import (
    DEFAULT_QC_MODELS,
    QcLevel,
    QcReport,
    qc_test,
)
from datagen.test_sessions.refine import refine_test
from lib.llm import GPT, model_scope  # type: ignore
from lib.tokens import make_bucket, record_stage  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge helper (safety net against refine LLM dropping required fields)
# ---------------------------------------------------------------------------

def _merge_refined(prev: dict, refined: dict) -> dict:
    """Merge refined over prev. Safety net against refine LLM dropping
    required fields:

      - `gold_value`: legitimately null (tool_identity / confirm-only),
        so only restore when the key is entirely absent.
      - `references` (top-level OR under local_env): restore from prev
        if refined dropped/emptied. Refine prompts emit `references` at
        the top level; older / mixed data may carry under `local_env`.

    Note: labels.task_success.required_actions is NOT merged here — it's
    derived fresh from rule + gold_value in stamp_session, so any stale
    value will be overwritten anyway.
    """
    out = dict(refined)

    # gold_value: null is legit; restore only if the key is missing.
    if "gold_value" not in out and "gold_value" in prev:
        out["gold_value"] = prev["gold_value"]

    # references: refine prompt now emits at top level; gen prompt does too.
    # Restore from prev (top-level or local_env) if refined dropped/emptied.
    refined_refs = out.get("references") or (out.get("local_env") or {}).get("references")
    if not refined_refs:
        prev_refs = prev.get("references") or (prev.get("local_env") or {}).get("references")
        if prev_refs:
            out["references"] = prev_refs

    return out


# ---------------------------------------------------------------------------
# Per-rule loop
# ---------------------------------------------------------------------------


def process_rule(
    rule: dict,
    persona_id: str,
    qc_level: QcLevel,
    gen_model: str,
    qc_models: tuple[str, ...] = DEFAULT_QC_MODELS,
    max_refine_rounds: int = CONFIG.test_max_refine_rounds,
    token_bucket: dict | None = None,
) -> tuple[dict | None, dict]:
    """Returns (accepted_session or None, audit_log_entry).

    Model knobs:
      - gen_model  : authoring (generate_candidate + refine_test). Typically
                     a strong instruction-follower (GPT).
      - qc_models  : sample list for oracle/passive trace simulation.
                     Each entry contributes one upper + one lower sample
                     (repeats of the same model = independent samples;
                     distinct models = ensemble across models).
                     Default (Gemini × 3) = 6 calls per FULL QC.
                     Upper aggregates best-of-all; lower worst-of-all.
    """
    # Re-enter the persona-level model_scope inside this worker thread —
    # ThreadPool workers don't inherit thread-local scope from the parent,
    # so the caller passes its bucket explicitly via `token_bucket`.
    cm = model_scope(token_bucket) if token_bucket is not None else contextlib.nullcontext()
    with cm:
        return _process_rule_inner(
            rule, persona_id, qc_level, gen_model, qc_models, max_refine_rounds,
        )


def _process_rule_inner(
    rule: dict,
    persona_id: str,
    qc_level: QcLevel,
    gen_model: str,
    qc_models: tuple[str, ...],
    max_refine_rounds: int,
) -> tuple[dict | None, dict]:
    rid = rule["rule_id"]
    try:
        candidate = generate_candidate(rule, persona_id, model=gen_model)
    except CandidateGenerationInfraError as e:
        return None, {
            "rule_id": rid,
            "passed": False,
            "reason": "gen_infra_failed",
            "error": str(e),
            "n_rounds": 0,
            "rounds": [],
            "infra_failed": True,
        }
    if candidate is None:
        return None, {
            "rule_id": rid,
            "passed": False,
            "reason": "gen_failed",
            "rounds": [],
        }

    rounds: list[dict] = []
    final_report: QcReport | None = None
    domain = domain_of_rule(rule) or "?"

    for refine_round in range(max_refine_rounds + 1):
        # QC does NOT take persona_brief — runner's test-phase agent has no
        # persona visibility, and the trace simulation must match.
        report = qc_test(candidate, rule, level=qc_level, qc_models=qc_models)
        final_report = report
        round_entry: dict = {
            "round_idx": refine_round,
            "candidate": candidate,
            "qc": report.to_dict(),
            "refined_candidate": None,
            "refine_failed": False,
        }
        rounds.append(round_entry)

        if report.passed:
            return candidate, {
                "rule_id": rid,
                "passed": True,
                "n_rounds": len(rounds),
                "rounds": rounds,
                "qc": report.to_dict(),
            }

        # LLM infra failure — refine would hit the same provider error and
        # burn more calls for nothing. Break out; surface as infra in the
        # dropped-items log so it's not mis-counted as a design failure.
        if report.llm_infra_failed:
            round_entry["infra_failed"] = True
            logger.warning(
                "[%s] %s LLM infra failure at QC — skipping refine. reasons=%s",
                persona_id, rid, report.failure_reasons,
            )
            break

        if refine_round == max_refine_rounds:
            break

        refined = refine_test(candidate, report, rule, model=gen_model)
        if refined is None:
            round_entry["refine_failed"] = True
            break
        # Safety net: refine LLM sometimes drops gold_value / references.
        # Merge refined over previous candidate — missing keys fall back to
        # previous values instead of getting lost. Re-stamp identity fields.
        merged = _merge_refined(candidate, refined)
        candidate = stamp_session(merged, rule, persona_id, domain)
        round_entry["refined_candidate"] = candidate

    return None, {
        "rule_id": rid,
        "passed": False,
        "n_rounds": len(rounds),
        "rounds": rounds,
        "qc": final_report.to_dict() if final_report else None,
        "infra_failed": bool(final_report and final_report.llm_infra_failed),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_persona(
    persona_id: str,
    rule_id_filter: str | None,
    qc_level: QcLevel,
    gen_model: str,
    qc_models: tuple[str, ...],
    max_refine_rounds: int,
    force: bool,
    rule_ids_exact: set[str] | None = None,
) -> dict:
    pdir = PERSONAS_DIR / persona_id
    rules_path = pdir / "rules.json"
    if not rules_path.exists():
        logger.error("[%s] missing rules.json", persona_id)
        return {"persona_id": persona_id, "status": "missing_inputs"}

    rules = read_json(rules_path)

    if rule_ids_exact:
        # exact-match multi-id filter (takes precedence over prefix)
        rules = [r for r in rules if r["rule_id"] in rule_ids_exact]
    elif rule_id_filter:
        rules = [r for r in rules if r["rule_id"].startswith(rule_id_filter)]

    out_dir = pdir / "test_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        before = len(rules)
        rules = [r for r in rules if not (out_dir / f"{r['rule_id']}.json").exists()]
        skipped = before - len(rules)
        if skipped:
            logger.info("[%s] skipped %d already-done (use --force)", persona_id, skipped)

    if not rules:
        logger.info("[%s] nothing to do", persona_id)
        return {"persona_id": persona_id, "status": "nothing_to_do"}

    logger.info(
        "[%s] processing %d rules, qc_level=%s, max_refine=%d, "
        "gen_model=%s, qc_models=%s (concurrency capped by llm.PER_MODEL_MAX_CONCURRENCY)",
        persona_id, len(rules), qc_level.value, max_refine_rounds,
        gen_model, list(qc_models),
    )

    stats = {"passed": 0, "dropped": 0, "infra_failed": 0, "rounds_total": 0}
    logs: list[dict] = []
    dropped_items: list[dict] = []
    t0 = time.time()

    # Persona-level token bucket shared across all process_rule workers and
    # nested QC ensemble workers. Each worker re-enters model_scope with
    # this same bucket reference; lib.llm._record_model_usage is lock-guarded
    # so concurrent updates are safe.
    token_bucket = make_bucket()

    # ThreadPoolExecutor sized to the task count — every rule is queued
    # immediately and threads block on the LLM-layer per-model semaphore
    # rather than competing for a small worker pool.
    with ThreadPoolExecutor(max_workers=max(1, len(rules))) as pool:
        futs = {
            pool.submit(process_rule, r, persona_id,
                        qc_level, gen_model, qc_models,
                        max_refine_rounds, token_bucket): r
            for r in rules
        }
        for fut in as_completed(futs):
            rule = futs[fut]
            rid = rule["rule_id"]
            try:
                session, log_entry = fut.result()
            except Exception as e:
                stats["dropped"] += 1
                dropped_items.append({"rule_id": rid, "error": str(e)})
                logger.error("[%s] %s crashed: %s", persona_id, rid, e)
                continue

            logs.append(log_entry)
            n_rounds = log_entry.get("n_rounds", 0)
            stats["rounds_total"] += n_rounds

            if session is not None:
                write_json(out_dir / f"{rid}.json", session)
                stats["passed"] += 1
                upper = log_entry["qc"].get("upper_matched")
                lower = log_entry["qc"].get("lower_matched")
                logger.info(
                    "[%s] %-40s PASS (rounds=%d, upper=%s, lower=%s)",
                    persona_id, rid, n_rounds, upper, lower,
                )
            else:
                infra = bool(log_entry.get("infra_failed"))
                if infra:
                    stats["infra_failed"] += 1
                else:
                    stats["dropped"] += 1
                dropped_item = {
                    "rule_id": rid,
                    "rounds": n_rounds,
                    "reasons": log_entry.get("qc", {}).get("failure_reasons", []),
                    "infra_failed": infra,
                }
                if log_entry.get("error"):
                    dropped_item["error"] = log_entry["error"]
                dropped_items.append(dropped_item)
                reasons = log_entry.get("qc", {}).get("failure_reasons") or [log_entry.get("reason", "?")]
                logger.warning(
                    "[%s] %-40s %s (rounds=%d) reasons=%s",
                    persona_id, rid,
                    "INFRA-FAIL" if infra else "DROP",
                    n_rounds, reasons[:2],
                )

    # Persist audit logs
    write_json(pdir / "test_session_qc_log.json", logs)
    if dropped_items:
        write_json(pdir / "test_session_dropped.json", dropped_items)

    # Merge token usage into the persona's token_usage.json.
    record_stage(pdir, "test_session", token_bucket)

    elapsed = time.time() - t0
    total = stats["passed"] + stats["dropped"] + stats["infra_failed"]
    avg_rounds = stats["rounds_total"] / max(1, total)
    logger.info(
        "[%s] done in %.1fs — passed=%d dropped=%d infra_failed=%d avg_rounds=%.1f",
        persona_id, elapsed,
        stats["passed"], stats["dropped"], stats["infra_failed"], avg_rounds,
    )
    return {"persona_id": persona_id, "status": "ok", **stats}


def main():
    parser = argparse.ArgumentParser(description="The test-session stage pipeline: gen→qc→refine for test sessions")
    parser.add_argument("--persona-id", type=str, required=True)
    parser.add_argument("--rule-id", type=str, default=None,
                        help="only process rules whose id starts with this")
    parser.add_argument("--rule-ids", type=str, default=None,
                        help="comma-separated exact rule_ids (overrides --rule-id)")
    parser.add_argument("--qc-level", type=str, default="full",
                        choices=["static", "oracle", "full"],
                        help="default full (upper+lower bound LLM QC)")
    parser.add_argument("--max-refine", type=int, default=CONFIG.test_max_refine_rounds)
    parser.add_argument("--model", type=str, default=GPT,
                        help="model for gen + refine (authoring); default GPT")
    parser.add_argument("--qc-models", type=str,
                        default=",".join(DEFAULT_QC_MODELS),
                        help=("comma-separated list of QC samples (one LLM call per "
                              "entry per bound; repeats of same model = independent "
                              "samples, distinct models = ensemble across models). "
                              f"default {','.join(DEFAULT_QC_MODELS)}"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    qc_models = tuple(m.strip() for m in args.qc_models.split(",") if m.strip())
    rule_ids_exact = None
    if args.rule_ids:
        rule_ids_exact = {s.strip() for s in args.rule_ids.split(",") if s.strip()}

    result = run_persona(
        args.persona_id,
        args.rule_id,
        QcLevel(args.qc_level),
        args.model,
        qc_models,
        args.max_refine,
        args.force,
        rule_ids_exact=rule_ids_exact,
    )
    if result.get("infra_failed", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
