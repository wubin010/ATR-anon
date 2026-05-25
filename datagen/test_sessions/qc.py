"""QC for a candidate TestSession.

Three layers (choose at top-level via QcLevel):
  - `static`: zero-LLM structural checks only (static_check.static_check).
  - `oracle`: static + upper-bound LLM trace (rule-injected agent SHOULD hit gold).
  - `full`:   static + upper + lower-bound (rule-NOT-injected agent should NOT hit gold).

Sampling (default Gemini × 3 parallel samples per bound):
  - upper = majority-of-ok: ≥2/3 ok samples reaching gold counts as hit
    (requires stable oracle behaviour — avoids counting single-sample
    lucky strikes as "oracle-can-solve")
  - lower = majority-of-ok: ≥2/3 ok passive samples reaching gold counts
    as leak (single hits are stochastic noise, especially for binary
    tool-identity rules where passive has ~50% per-sample hit rate)
  Total LLM calls per FULL QC = 3 samples × 2 bounds = 6 calls.

Adapted to the TestSession schema:
  - session_type "test" (was "payoff")
  - rule_ref carries canonical_answer (unified user-voice statement)
  - 4 flat check_type branches (tool_identity / param_id / param_enum /
    confirm); confirm rules emit a single get_user_confirmation call
    (no chain trailing the mutate)

The only LLM prompt used here is `prompts/qc.md` — the same "trace generation"
prompt used for both upper and lower bounds (toggled via RULE_CONTEXT_BLOCK).

QC ↔ Runtime alignment
======================

QC is a **constructibility filter**, not a runtime predictor. It certifies
that a test session has both:
  · Solvability: a rule-aware LLM with help can ground canonical_answer
    to gold action (upper bound passes)
  · Non-leakage: a rule-blind LLM with no preference context cannot
    trivially hit gold from instruction/refs alone (lower bound misses)

The byte-equivalent contract on the `<context>` block content (NOT the
surrounding instructions) holds:

    QC upper canonical_answer payload  ←byte-equivalent→  runtime Oracle (Targeted)

Five divergences exist between QC and runtime. One must-align (structural)
+ four by-design:

  1. canonical_answer injection payload — MUST ALIGN.
     QC upper goes through `ContextLayer.inject_prior_user_statement`
     and wraps with `<context>...</context>` exactly like
     run_episode.py → prompt_builder.py. Enforced by
     tests/test_qc_runtime_alignment.py.

  2. surrounding LLM instructions — BY-DESIGN ASYMMETRIC (Path B).
     QC prompt explicitly tells the LLM how to interpret <context>
     ("treat 'I always X' patterns as standing rules; actively apply").
     Runtime agent prompt uses a softer, less prescriptive scaffolding
     ("scan <context> for long-term preferences"). This makes QC a
     looser constructibility filter — sessions where the canonical_answer
     is in-principle groundable but require LLM to recognize standing-
     rule syntax pass QC, even if today's runtime LLM might not always
     recognize them. Trade-off: smaller "QC pass implies runtime pass"
     guarantee, larger benchmark longevity (filter doesn't bake in
     current LLM's recognition capability).

  3. references view — BY-DESIGN.
     QC shows refs verbatim in prompt; runner agent must call
     search_/list_ to discover them. Cost optimization.
     Discoverability gap is closed by static_check section 9
     (instruction-ref keyword consistency).

  4. persona / LS transcript — BY-DESIGN.
     QC has no persona; runtime Oracle also has none (skips LS phase),
     but runtime Passive accumulates LS transcript in <context>. So
     QC lower bound = "passive lower bound when LS leak = 0". Runtime
     Passive's actual hit rate ≥ QC lower hit rate (LS may leak some
     rule direction). This is exactly what the ATR experiment measures:
     ATR must outperform passive-with-LS-leak.

  5. LLM model — BY-DESIGN.
     QC uses Gemini × 3 (cheap, multi-sample for stability). Runtime
     uses whatever the experiment's agent model is.

Implication: oracle absolute hit rate at runtime may
be lower than QC upper hit rate (because runtime LLM, without QC's
explicit standing-rule scaffolding, sometimes fails to apply an
in-principle-applicable rule). This is the "agent capability deficit"
gap and is itself a finding to report. Variant comparisons (ATR vs
Passive gap) are unaffected — all runtime variants share the same
softer scaffolding, so they're measured on a level field.

Sanity-check thresholds applied after frozen (1-persona end-to-end):
  - Oracle (Targeted) pass rate ≥ 70%   (was 90%; relaxed because
    asymmetric scaffolding lets some marginal-recognition rules through)
  - Passive pass rate ≤ 30%
  - ATR pass rate strictly between passive and oracle
  Fail any → datagen regression, not framework regression.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datagen._common import (
    _tool_to_openai_schema,
    fill_prompt,
    load_prompt,
    to_json,
    tool_map,
)
from datagen.test_sessions.static_check import static_check
from lib.llm import GEMINI, call_llm_json, current_model_bucket, model_scope  # type: ignore

# Default QC sampler: Gemini × 3 parallel samples. Three independent
# trace draws from the same model reduce single-sample stochasticity.
# Each entry contributes one sample per bound → 6 total LLM calls per
# FULL QC (upper × 3 + lower × 3).
DEFAULT_QC_MODELS: tuple[str, ...] = (GEMINI, GEMINI, GEMINI)

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants mirrored from static_check.py
# ---------------------------------------------------------------------------

# Enum-value "leakage" detection now lives in static_check (instruction-text
# scanning). It used to also drive per-arg strict equality here, but that's
# redundant now — scalars are already strict-equal by default in
# evaluator._scalar_or_list_match. Free-text arg skipping is deliberately
# removed: it loosened QC below runtime strictness, producing "QC pass /
# runtime fail" false positives.

# ---------------------------------------------------------------------------
# Level + Report
# ---------------------------------------------------------------------------

class QcLevel(str, Enum):
    STATIC = "static"
    ORACLE = "oracle"  # static + upper
    FULL = "full"      # static + upper + lower


@dataclass
class QcReport:
    passed: bool
    level: QcLevel
    n_samples: int = 1  # total samples per bound = len(qc_models)
    qc_models: list[str] = field(default_factory=list)
    static_errors: list[str] = field(default_factory=list)

    # Aggregated (best-of-all for upper, worst-of-all for lower)
    upper_matched: bool | None = None
    lower_matched: bool | None = None
    upper_hits: int | None = None      # m of N
    lower_hits: int | None = None

    # Representative single sample (first matched if any, else first sample).
    upper_trace: list[dict] | None = None
    lower_trace: list[dict] | None = None
    upper_reason: str | None = None
    lower_reason: str | None = None

    # Full per-sample data — refine reads these to reason across samples.
    # Each list aligned with qc_models: upper_traces[i] is the trace produced
    # by qc_models[i] on the upper-bound call.
    upper_traces: list[list[dict]] | None = None
    lower_traces: list[list[dict]] | None = None
    upper_reasons: list[str] | None = None
    lower_reasons: list[str] | None = None
    upper_match_per_sample: list[bool] | None = None
    lower_match_per_sample: list[bool] | None = None

    # Prompt + raw response of the representative sample only.
    upper_prompt: str | None = None
    upper_raw_response: str | None = None
    lower_prompt: str | None = None
    lower_raw_response: str | None = None

    # Per-sample LLM infra errors (None = ok, string = diagnostic). Used
    # to exclude infra-failed samples from upper/lower aggregation so a
    # 402 / 5xx / parse failure doesn't masquerade as a business failure.
    upper_errors: list[str | None] | None = None
    lower_errors: list[str | None] | None = None

    # True when all samples on at least one bound failed at the LLM layer
    # — the session couldn't be evaluated, so the failure is infra, not
    # design. Pipeline uses this to skip refine (refine would hit the
    # same infra issue and burn more calls for nothing).
    llm_infra_failed: bool = False

    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "level": self.level.value,
            "n_samples": self.n_samples,
            "qc_models": self.qc_models,
            "static_errors": self.static_errors,
            "upper_matched": self.upper_matched,
            "lower_matched": self.lower_matched,
            "upper_hits": self.upper_hits,
            "lower_hits": self.lower_hits,
            "upper_reason": self.upper_reason,
            "lower_reason": self.lower_reason,
            "upper_trace": self.upper_trace,
            "lower_trace": self.lower_trace,
            "upper_traces": self.upper_traces,
            "lower_traces": self.lower_traces,
            "upper_reasons": self.upper_reasons,
            "lower_reasons": self.lower_reasons,
            "upper_match_per_sample": self.upper_match_per_sample,
            "lower_match_per_sample": self.lower_match_per_sample,
            "upper_prompt": self.upper_prompt,
            "upper_raw_response": self.upper_raw_response,
            "lower_prompt": self.lower_prompt,
            "lower_raw_response": self.lower_raw_response,
            "upper_errors": self.upper_errors,
            "lower_errors": self.lower_errors,
            "llm_infra_failed": self.llm_infra_failed,
            "failure_reasons": self.failure_reasons,
        }


# ---------------------------------------------------------------------------
# LLM trace generation (upper / lower)
# ---------------------------------------------------------------------------


_UPPER_GUIDANCE = (
    "The block below contains a long-term preference (a \"standing rule\") "
    "this user has expressed in their own first-person voice. "
    "Apply this preference to the current task whenever it is relevant — "
    "let it shape the tools and arguments you choose."
)


def _build_rule_context_block(canonical_answer: str, rule_injected: bool) -> str:
    """Build the rule-context block injected into the QC agent prompt.

    Upper bound (rule_injected=True): a guidance sentence followed by the
    `<context>...</context>` block. The `<context>` block reuses runner's
    `ContextLayer`, so its inner content is byte-identical with what
    runner's Oracle (Targeted) sees in its own `<context>` block:

      runner Oracle path (run_episode.py:284-297):
        per_test_layer = CrossSessionLayer.create("context", ...)
        per_test_layer.inject_prior_user_statement(canonical_answer)
        → at session entry, prompt_builder wraps
          layer.get_context_string() in `<context>...</context>`.

      QC upper path (this function):
        same ContextLayer construction → wrap with same `<context>` tag,
        plus an upper-only guidance sentence prepended.

    The QC-only guidance prepended on upper is intentional: QC is a
    constructibility filter, not a runtime predictor (see module
    docstring's QC-vs-Runtime alignment section). Telling the upper-bound LLM
    explicitly that the `<context>` content is a standing rule to apply
    only makes QC a *looser* check than runtime ("if oracle in QC can't
    apply this rule even with the explicit prompt, the session is
    structurally broken"). Runtime Oracle's softer prompting is then
    measured separately as agent capability, not data-construction
    quality. Test alignment now requires the runner `<context>` block
    to be byte-equivalent *contained in* the QC upper output — see
    tests/test_qc_runtime_alignment.py.

    Lower bound (rule_injected=False) returns empty — runner passive has
    no variant block at all (prompt_builder.py:280). Adding any
    "zero-knowledge / don't speculate" framing would suppress the QC LLM
    below what runner passive naturally does, producing false-negative
    lower_matched.

    Known by-design divergence vs runner (see module docstring QC ↔
    Runtime alignment): references short-circuit + persona/LS transcript
    absence. Both are documented and bounded.
    """
    if not rule_injected:
        return ""
    layer = _ContextLayer()
    layer.inject_prior_user_statement(canonical_answer)
    inner = layer.get_context_string() or ""
    return f"{_UPPER_GUIDANCE}\n\n<context>\n{inner}\n</context>"


def _render_references(refs: list[dict]) -> str:
    if not refs:
        return "(empty)"
    lines = []
    for r in refs:
        attrs = json.dumps(r.get("attributes", {}), ensure_ascii=False)
        lines.append(f"- id={r.get('id')}, type={r.get('type')}, attributes={attrs}")
    return "\n".join(lines)


def llm_qc_trace(
    session: dict,
    rule_injected: bool,
    rule: dict,
    model: str = GEMINI,
) -> tuple[list[dict], str, str, str | None]:
    """Run the LLM as an agent, return (trace, prompt, raw_response_json, error).

    `error` is None on success; a short diagnostic string when the LLM
    call itself failed or returned something unparseable. Callers must
    NOT treat an empty trace + error!=None as "agent chose nothing" —
    that's infra failure (e.g. 402 quota / 5xx / malformed JSON), not
    a business signal.

    Matches runner's test-phase visibility contract: agent sees the
    instruction + allowed tools + references (references only shown here
    because QC short-circuits the search/list discovery step). Persona is
    intentionally NOT passed — in runner's test phase, agent has no persona
    context; it only obtains user info through the learning-phase user_sim,
    which is offline at test time.
    """
    whitelist = set(session.get("local_env", {}).get("tools") or [])
    tmap = tool_map()
    tool_specs = [
        _tool_to_openai_schema(tmap[t])
        for t in sorted(whitelist) if t in tmap
    ]

    template = load_prompt(PROMPTS_DIR, name="qc")
    prompt = fill_prompt(
        template,
        INSTRUCTION=session.get("instruction", ""),
        TOOL_SPECS=to_json(tool_specs),
        REFERENCES=_render_references(session.get("local_env", {}).get("references") or []),
        RULE_CONTEXT_BLOCK=_build_rule_context_block(
            rule.get("canonical_answer", ""),
            rule_injected=rule_injected,
        ),
    )

    try:
        result = call_llm_json(prompt, model=model, temperature=0.2, max_retries=3)
    except Exception as e:
        logger.warning("llm_qc_trace LLM call failed (model=%s): %s", model, e)
        return [], prompt, to_json({"error": str(e)}), f"llm_call_failed: {e}"

    if not isinstance(result, dict):
        return [], prompt, to_json({"error": "non_dict_response", "value": result}), "non_dict_response"

    trace = result.get("trace")
    if not isinstance(trace, list):
        return [], prompt, to_json(result), "no_trace_field"

    cleaned: list[dict] = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        cleaned.append({
            "tool": step.get("tool"),
            "arguments": step.get("arguments") or {},
        })
    return cleaned, prompt, to_json(result), None


# ---------------------------------------------------------------------------
# Trace match — delegated to evaluator.action_match for single source of
# truth with runtime task_success judgement. QC must never drift from
# runtime match semantics, otherwise "QC pass" stops implying "runtime
# pass" (the whole point of QC is to preview runtime decisions).
# ---------------------------------------------------------------------------

# Import evaluator.match_actions (runner-side schema) — we build a minimal
# fake SessionTrajectory so evaluator can do the lifting.
import sys as _sys
from pathlib import Path as _Path

_RUNNER_DIR = _Path(__file__).resolve().parent.parent.parent / "runner"
_EVALUATOR_DIR = _Path(__file__).resolve().parent.parent.parent / "evaluator"
for _d in (_RUNNER_DIR, _EVALUATOR_DIR):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
from schemas import Message as _Message  # type: ignore  # noqa: E402
from schemas import RequiredAction as _RequiredAction  # type: ignore  # noqa: E402
from schemas import SessionTrajectory as _SessionTrajectory  # type: ignore  # noqa: E402
from schemas import ToolCall as _ToolCall  # type: ignore  # noqa: E402
from action_match import match_actions as _match_actions  # type: ignore  # noqa: E402
from memory import ContextLayer as _ContextLayer  # type: ignore  # noqa: E402


def trace_matches_gold(
    pred: list[dict],
    gold: list[dict],
) -> bool:
    """Return True iff the predicted trace satisfies the gold required_actions
    under the same semantics runtime uses for task_success.

    Delegates to `evaluator.action_match.match_actions`. Under the single-
    axis schema gold is always len=1 (Shape A direct or Shape B confirm
    truncated).

    Wrapping dict → pydantic: pred becomes one assistant Message with N
    tool_calls; gold becomes a list of RequiredAction.
    """
    calls = [
        _ToolCall(
            id=f"qc_c{i}",
            name=(step.get("tool") or ""),
            arguments=step.get("arguments") or {},
        )
        for i, step in enumerate(pred)
    ]
    traj = _SessionTrajectory(
        session_id="qc_synthetic",
        session_type="test",
        # QC synthetic trajectories only need a valid variant string;
        # `oracle_target` is the oracle variant.
        agent_variant="oracle_target",
        messages=[_Message(role="assistant", turn_idx=0, tool_calls=calls)],
    )
    required = [
        _RequiredAction(
            tool=g["tool"],
            arguments=g.get("arguments") or {},
            compare_args=g.get("compare_args"),
        )
        for g in gold
    ]
    step_results = _match_actions(traj, required)
    return bool(step_results) and all(sr.passed for sr in step_results)


# ---------------------------------------------------------------------------
# Top-level QC entry
# ---------------------------------------------------------------------------


def _extract_reason(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return (parsed.get("reason") or "").strip()
    except Exception:
        pass
    return ""


def _upper_failure_msg(
    reasons: list[str],
    errors: list[str | None],
    hits: int,
    n_ok: int,
    n_total: int,
) -> str:
    """Build failure message for upper bound.

    hits / n_ok count only LLM-ok samples; infra-failed samples are
    excluded (they can't tell us anything about oracle behaviour). The
    message calls out infra-excluded samples so refine knows the
    denominator shrank.
    """
    base = (
        f"upper_missed_gold ({hits}/{n_ok}): oracle agent did not reach gold "
        f"in {n_ok - hits} of {n_ok} evaluable samples"
    )
    if n_ok < n_total:
        base += f" ({n_total - n_ok}/{n_total} samples failed at LLM layer, excluded)"
    quoted = [f'"{r}"' for r, e in zip(reasons, errors) if r and e is None]
    if quoted:
        base += f". Oracle samples reasons: [{', '.join(quoted)}]"
    return base


def _lower_failure_msg(
    reasons: list[str],
    matches: list[bool],
    errors: list[str | None],
    hits: int,
    n_ok: int,
    n_total: int,
) -> str:
    """Build failure message for lower bound. Same ok-vs-total handling
    as _upper_failure_msg.
    """
    base = (
        f"lower_matched_gold ({hits}/{n_ok}): passive agent hit gold in "
        f"{hits} of {n_ok} evaluable samples — instruction or references "
        "leak rule direction"
    )
    if n_ok < n_total:
        base += f" ({n_total - n_ok}/{n_total} samples failed at LLM layer, excluded)"
    hit_reasons = [
        f'"{r}"' for r, m, e in zip(reasons, matches, errors)
        if m and r and e is None
    ]
    if hit_reasons:
        base += f". Passive hit-reasons: [{', '.join(hit_reasons)}]"
    base += (
        ". Identify what in the instruction or references makes the gold "
        "feel like the natural choice for a passive agent — strip "
        "direction-revealing wording, weaken gold's pull, or strengthen the "
        "decoys' pull."
    )
    return base


def _auto_upgrade_level(level: QcLevel, rule: dict) -> QcLevel:
    """Auto-upgrade ORACLE → FULL for tool_identity rules (need to verify
    passive doesn't accidentally pick the gold tool). Pipeline default is
    already FULL, so this branch only fires for explicit --qc-level=oracle
    invocations.
    """
    if level != QcLevel.ORACLE:
        return level
    if rule.get("check_type") == "tool_identity":
        return QcLevel.FULL
    return level


def _run_ensemble(
    session: dict,
    rule_injected: bool,
    rule: dict,
    models: list[str],
) -> tuple[list[list[dict]], list[str], list[str], list[str], list[str | None]]:
    """Run llm_qc_trace once per entry in `models`, concurrently. Returns
    (traces, prompts, raws, reasons, errors) each aligned with `models`
    by index. errors[i] is None when sample i succeeded, or a diagnostic
    string when the LLM call failed / response was unparseable. Callers
    MUST filter out error samples before aggregating upper/lower hit
    counts, otherwise infra failures get mis-attributed to business
    failures.
    """
    # Snapshot the persona-level model_scope bucket from the parent thread
    # so each ensemble worker can re-enter it (thread-local stacks don't
    # propagate across ThreadPool boundaries).
    parent_bucket = current_model_bucket()

    def _one_for(m: str) -> tuple[list[dict], str, str, str | None]:
        if parent_bucket is None:
            return llm_qc_trace(session, rule_injected, rule, m)
        with model_scope(parent_bucket):
            return llm_qc_trace(session, rule_injected, rule, m)

    if len(models) == 1:
        results = [_one_for(models[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(models)) as ex:
            results = list(ex.map(_one_for, models))

    traces, prompts, raws, errors = zip(*results)
    reasons = [_extract_reason(r) for r in raws]
    return list(traces), list(prompts), list(raws), reasons, list(errors)


def _representative_index(
    matches: list[bool],
    errors: list[str | None],
    prefer_match: bool,
) -> int:
    """Pick a representative sample for QcReport.upper_trace / lower_trace.
    Prefer in order: ok+matched (shows success/leak pattern), any ok
    sample (clean trace), then index 0 (fallback when all failed).
    """
    if prefer_match:
        for i, (m, e) in enumerate(zip(matches, errors)):
            if m and e is None:
                return i
    for i, e in enumerate(errors):
        if e is None:
            return i
    return 0




def qc_test(
    session: dict,
    rule: dict,
    level: QcLevel = QcLevel.FULL,
    qc_models: tuple[str, ...] = DEFAULT_QC_MODELS,
) -> QcReport:
    """Run QC per level using N parallel samples.

    qc_models: tuple of model names, one trace per entry per bound.
      Typically N repeats of the same model (e.g. Gemini × 3) for
      stochasticity reduction; can also be distinct models for ensemble
      diversity.
      - upper aggregated as majority-of-ok (≥2/3 of ok samples → matched)
      - lower aggregated as majority-of-ok (≥2/3 of ok samples → matched)
      Default is (GEMINI, GEMINI, GEMINI): 6 LLM calls per FULL QC
      (upper × 3 + lower × 3).

    Persona context is intentionally NOT passed to QC — runner's test-phase
    agent has no persona visibility (see llm_qc_trace docstring).
    """
    effective_level = _auto_upgrade_level(level, rule)
    if effective_level != level:
        logger.debug("[%s] auto-upgraded QC %s → %s",
                     rule.get("rule_id"), level.value, effective_level.value)

    models = list(qc_models)
    if not models:
        raise ValueError("qc_models must be non-empty")

    static_errors = static_check(session, rule)

    report = QcReport(
        passed=False,
        level=effective_level,
        n_samples=len(models),
        qc_models=list(models),
        static_errors=static_errors,
    )

    if static_errors:
        report.failure_reasons = list(static_errors)
        return report

    if effective_level == QcLevel.STATIC:
        report.passed = True
        return report

    gold = session["labels"]["task_success"]["required_actions"]
    n_total = len(models)

    # ── Upper bound: one sample per entry, rule injected → ANY ok hit = match ─
    upper_traces, upper_prompts, upper_raws, upper_reasons, upper_errors = \
        _run_ensemble(session, rule_injected=True, rule=rule, models=models)

    upper_matches = [trace_matches_gold(t, gold) for t in upper_traces]
    upper_ok_idx = [i for i, e in enumerate(upper_errors) if e is None]
    upper_n_ok = len(upper_ok_idx)
    # Only ok samples count toward hits/matched — infra-failed samples
    # can't tell us anything about oracle behaviour.
    upper_hits = sum(upper_matches[i] for i in upper_ok_idx)
    # Majority-of-ok: need ≥2/3 of ok samples to hit. At n_ok=3 that's ≥2;
    # at n_ok=1 it collapses to "must hit". Pure any-hit would flag 1/3
    # lucky gemini runs as "oracle-can-solve" and over-count passing
    # sessions that are actually unstable at runtime.
    upper_matched = upper_n_ok > 0 and upper_hits * 3 >= upper_n_ok * 2

    rep_idx_u = _representative_index(upper_matches, upper_errors, prefer_match=True)
    report.upper_traces = upper_traces
    report.upper_reasons = upper_reasons
    report.upper_errors = upper_errors
    report.upper_match_per_sample = upper_matches
    report.upper_hits = upper_hits
    report.upper_matched = upper_matched
    report.upper_trace = upper_traces[rep_idx_u]
    report.upper_reason = upper_reasons[rep_idx_u]
    report.upper_prompt = upper_prompts[rep_idx_u]
    report.upper_raw_response = upper_raws[rep_idx_u]

    # Short-circuit: all upper samples hit LLM infra failure. This is
    # NOT a design problem — refine would hit the same error and burn
    # more calls for nothing. Flag it so the pipeline can skip refine.
    if upper_n_ok == 0:
        distinct = sorted({e for e in upper_errors if e})
        report.llm_infra_failed = True
        report.failure_reasons.append(
            f"llm_infra_error_all_upper_samples: {distinct}"
        )
        return report

    if effective_level == QcLevel.ORACLE:
        if upper_matched:
            report.passed = True
        else:
            report.failure_reasons.append(
                _upper_failure_msg(
                    upper_reasons, upper_errors, upper_hits, upper_n_ok, n_total,
                )
            )
        return report

    # ── Lazy eval: if upper already failed, skip lower entirely ───────────
    # passed = upper_matched AND NOT lower_matched. If upper_matched=False,
    # session can never pass, so running lower is pure waste. This saves
    # 3 gemini calls per upper-missed rule (~40% of QC cost on fail path).
    if not upper_matched:
        report.failure_reasons.append(
            _upper_failure_msg(
                upper_reasons, upper_errors, upper_hits, upper_n_ok, n_total,
            )
        )
        return report

    # ── FULL: lower bound, one sample per entry, ≥2/3 ok hit → leak ───────
    lower_traces, lower_prompts, lower_raws, lower_reasons, lower_errors = \
        _run_ensemble(session, rule_injected=False, rule=rule, models=models)

    lower_matches = [trace_matches_gold(t, gold) for t in lower_traces]
    lower_ok_idx = [i for i, e in enumerate(lower_errors) if e is None]
    lower_n_ok = len(lower_ok_idx)
    lower_hits = sum(lower_matches[i] for i in lower_ok_idx)
    # Majority-of-ok: need ≥2/3 of ok passive samples to hit before we
    # call it a leak. A single lucky hit is stochastic noise — e.g. for
    # a binary tool-identity rule, passive has 50% per-sample hit rate
    # and P(≥1 hit in 3) ≈ 87.5%, which would false-positive most
    # sessions under the old any-hit rule.
    lower_matched = lower_n_ok > 0 and lower_hits * 3 >= lower_n_ok * 2

    rep_idx_l = _representative_index(lower_matches, lower_errors, prefer_match=True)
    report.lower_traces = lower_traces
    report.lower_reasons = lower_reasons
    report.lower_errors = lower_errors
    report.lower_match_per_sample = lower_matches
    report.lower_hits = lower_hits
    report.lower_matched = lower_matched
    report.lower_trace = lower_traces[rep_idx_l]
    report.lower_reason = lower_reasons[rep_idx_l]
    report.lower_prompt = lower_prompts[rep_idx_l]
    report.lower_raw_response = lower_raws[rep_idx_l]

    if lower_n_ok == 0:
        distinct = sorted({e for e in lower_errors if e})
        report.llm_infra_failed = True
        report.failure_reasons.append(
            f"llm_infra_error_all_lower_samples: {distinct}"
        )
        return report

    # We only reach here when upper_matched is True (lazy eval above
    # returned early otherwise), so only lower_matched can flip the
    # verdict.
    if lower_matched:
        report.failure_reasons.append(
            _lower_failure_msg(
                lower_reasons, lower_matches, lower_errors,
                lower_hits, lower_n_ok, n_total,
            )
        )
    report.passed = not lower_matched
    return report
