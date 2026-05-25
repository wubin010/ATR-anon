"""Single source of truth for datagen pipeline numeric knobs.

Stages import `CONFIG` from here instead of hardcoding per-stage defaults.
Changing a knob means editing this file once, not hunting through 6 stages.

CLI flags still act as per-invocation overrides (e.g. `--n 30` beats the
config default), but the canonical defaults live here.

Concurrency: datagen stage CLIs no longer expose `--workers`. Stage
ThreadPools are sized to task count so every task is queued immediately;
actual request concurrency is gated by `PER_MODEL_MAX_CONCURRENCY` in
`src/llm.py` — the single choke point.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatagenConfig:
    # ── The rule stage · rules (gen → qc) ─────────────────────────────────
    # LLM-authored rules per persona. After rule QC drops default-heavy /
    # weak-wedge rules (~20-50% drop rate, persona-dependent) and the
    # test-session stage QC drops un-executable / lower-leak test sessions
    # (~15-20% drop rate), downstream typically yields 11-17 ts/persona — a
    # comfortable 10-20 band for benchmarking. Empirically 24-30 is the sweet
    # spot; raising N further mostly produces marginal rules that rule QC drops.
    rule_count_per_persona: int = 24

    # ── The learning-session skeleton stage · session_gen/skeleton ─────────
    # Number of learning-session skeletons per persona is derived from the
    # persona's accepted-rule count (len(rules_qc.json)) so more rules →
    # more learning surface for the ATR agent to land on:
    #
    #     N = round(rule_count * learning_multiplier) + learning_drop_buffer
    #     clamped to [learning_session_min, learning_session_max]
    #
    # The multiplier sizes the candidate pool, not the final episode
    # trajectory. Episode compose later selects K = test_sessions × 2. Keeping
    # this pool at ~3× leaves room for learning-session fill rejects, pure-noise
    # fill, and signal coverage choices without excessive fill cost.
    #
    # When rules_qc.json is missing (learning-session skeleton run ahead of
    # rule QC or standalone), fall back to rule_count_per_persona as the
    # rule-count proxy.
    learning_multiplier: float = 3.0
    learning_drop_buffer: int = 2
    learning_session_min: int = 8
    learning_session_max: int = 100

    # ── The test-session stage · test_session_gen/pipeline ─────────────────
    # Refine rounds AFTER gen. 2 is the empirical sweet spot:
    # - 0 (plan-then-generate only) drops 18-25% on first attempt
    # - 1 round saves ~half of those (lower_leak / static fixable cases)
    # - 2 rounds catches a few more (refine LLM converges)
    # - 3+ has marginal return: refine usually can't recover gen's
    #   underlying gold-ref mistakes after 2 attempts.
    test_max_refine_rounds: int = 2


CONFIG = DatagenConfig()
