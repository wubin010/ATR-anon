"""The rule stage pipeline · rules: gen → qc.

Two sub-stages: one batch generation pass (gen) followed by one batch
filter pass (qc):

  · rule generation — recall + bind in one LLM call. From persona.structured,
                  surface long-term rules with the full (check_type,
                  action_step) binding attached. Each action_step is a
                  single dict {tool, param} — no list, no chain.
  · rule QC      — batch tag QC across 7 cross-rule + per-rule criteria;
                  sees the bound rules and tags each keep/remove (1 LLM
                  call/persona)

Data flow per persona:

    structured.json
        │  gen   (LLM, full ontology in scope, recall + bind in one pass)
        ▼
    rules_gen.json     (bindable + LLM-marked unbindable mixed)
        │  qc    (LLM tag — sees rule + binding)
        ▼
    rules.json         (final — downstream reads this; bindable only)
                       (+ rules_qc_dropped.json, rules_gen_rejected.json)

CLI:
    uv run python -m datagen.rules.pipeline \\
        --persona-id <uuid_prefix> [--n 30] [--force] \\
        [--only gen|qc]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from datagen._common import PERSONAS_DIR
from datagen.config import CONFIG
from datagen.rules import gen, qc
from lib.llm import GPT, model_scope  # type: ignore
from lib.tokens import make_bucket, record_stage  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_persona_all(
    persona_id: str,
    n: int,
    model: str,
    force: bool,
    only: str | None = None,
) -> dict:
    """Run gen → qc for one persona. `only` skips the other.

    Both sub-stages share one `rule_gen` token bucket — they run
    sequentially in this thread, so a single `model_scope` covers both.
    Bucket gets merged into the persona's token_usage.json at the end.
    """
    results: dict = {"persona_id": persona_id}
    bucket = make_bucket()

    with model_scope(bucket):
        if only in (None, "gen"):
            r = gen.run_persona(persona_id, n=n, model=model, force=force)
            results["gen"] = r
            if r.get("status") not in ("ok", "skipped"):
                record_stage(PERSONAS_DIR / persona_id, "rule_gen", bucket)
                return results

        if only in (None, "qc"):
            r = qc.run_persona(persona_id, model=model, force=force)
            results["qc"] = r

    record_stage(PERSONAS_DIR / persona_id, "rule_gen", bucket)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="The rule stage pipeline (gen → qc)"
    )
    parser.set_defaults()
    parser.add_argument("--persona-id", type=str, default=None,
                        help="prefix match; if omitted, process all personas")
    parser.add_argument("--n", type=int, default=CONFIG.rule_count_per_persona,
                        help=f"target rule count (default {CONFIG.rule_count_per_persona})")
    parser.add_argument("--model", type=str, default=GPT)
    parser.add_argument("--only", type=str, default=None,
                        choices=["gen", "qc"],
                        help="run only one stage (skip the other)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not PERSONAS_DIR.exists():
        parser.error(f"personas dir not found: {PERSONAS_DIR}")

    all_pids = sorted(
        p.name for p in PERSONAS_DIR.iterdir()
        if p.is_dir() and (p / "structured.json").exists()
    )
    if args.persona_id:
        all_pids = [p for p in all_pids if p.startswith(args.persona_id)]
        if not all_pids:
            parser.error(f"no persona matched --persona-id {args.persona_id!r}")

    if not all_pids:
        logger.info("nothing to do.")
        return

    logger.info(
        "processing %d personas, model=%s, stage=%s "
        "(concurrency capped by llm.PER_MODEL_MAX_CONCURRENCY)",
        len(all_pids), args.model,
        args.only or "gen→qc",
    )
    with ThreadPoolExecutor(max_workers=max(1, len(all_pids))) as pool:
        futs = {
            pool.submit(
                run_persona_all, pid, args.n, args.model, args.force, args.only,
            ): pid
            for pid in all_pids
        }
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                result = fut.result()
                logger.info("[%s] done: %s", pid, result)
            except Exception as e:
                logger.error("[%s] failed: %s", pid, e, exc_info=True)


if __name__ == "__main__":
    main()
