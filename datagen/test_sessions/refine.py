"""Refine a failing TestSession candidate via LLM.

Takes the current candidate + QC failure bundle and asks the LLM to produce
a corrected version.

Failure bundle gives the LLM the oracle/passive self-reported reasons
(from QC) so it can reason about *why* the previous version was wrong,
not just *what* failed.
Same few-shot pool as `gen.py` is injected.

Persona is intentionally NOT in the prompt — same rationale as gen.py: the
runner's test-phase agent has no persona visibility.
"""
from __future__ import annotations

import logging
from pathlib import Path

from datagen._common import (
    fill_prompt,
    load_prompt,
    render_domain_tools,
    render_ref_schema,
    to_json,
)
from datagen.test_sessions.pool import load_pool_exemplars, render_exemplars_md
from datagen.test_sessions.qc import QcReport
from lib.llm import GPT, call_llm_json  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"

logger = logging.getLogger(__name__)


def refine_test(
    session: dict,
    report: QcReport,
    rule: dict,
    model: str = GPT,
) -> dict | None:
    """Return a revised session dict or None on LLM failure.

    Caller (pipeline) is responsible for re-stamping identity fields after
    refinement via `stamp_session`.
    """
    domain = session.get("domain") or "?"

    failure_bundle = {
        "n_samples": report.n_samples,
        "static_errors": report.static_errors,
        "upper_matched": report.upper_matched,
        "lower_matched": report.lower_matched,
        "upper_hits": report.upper_hits,
        "lower_hits": report.lower_hits,
        # Per-sample data so refine can reason across the N samples instead of
        # guessing from a single trace. upper_reasons / lower_reasons are the
        # LLM's self-reported decision rationale per sample (most actionable
        # signal); upper_traces / lower_traces are the actual tool calls.
        "upper_reasons": report.upper_reasons,
        "lower_reasons": report.lower_reasons,
        "upper_traces": report.upper_traces,
        "lower_traces": report.lower_traces,
        "upper_match_per_sample": report.upper_match_per_sample,
        "lower_match_per_sample": report.lower_match_per_sample,
        # Per-sample infra errors — samples with non-null error are LLM
        # failures, not business signal; refine should ignore them.
        "upper_errors": report.upper_errors,
        "lower_errors": report.lower_errors,
        "failure_reasons": report.failure_reasons,
    }

    exemplars = load_pool_exemplars(rule, max_count=3)

    # Strip stage-internal diagnostic fields before showing to the refine LLM.
    # Mirrors gen.py — _qc carries verdict tags that could bias refine output.
    rule_for_prompt = {
        k: v for k, v in rule.items() if k != "_qc"
    }

    template = load_prompt(PROMPTS_DIR, name="refine")
    prompt = fill_prompt(
        template,
        CURRENT_TASK=to_json(session),
        FAILURE_REASONS=to_json(failure_bundle),
        RULE_JSON=to_json(rule_for_prompt),
        DOMAIN=domain,
        DOMAIN_TOOL_SPECS=render_domain_tools(domain),
        REF_SCHEMA=render_ref_schema(domain),
        FEW_SHOT_EXAMPLES=render_exemplars_md(exemplars),
    )

    try:
        result = call_llm_json(prompt, model=model, temperature=0.3, max_retries=3)
    except Exception as e:
        logger.warning("refine_test LLM call failed: %s", e)
        return None

    if not isinstance(result, dict):
        logger.warning("refine_test returned non-dict: %s", type(result).__name__)
        return None

    return result
