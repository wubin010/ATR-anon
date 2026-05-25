"""Learning-session skeleton: session skeleton (thin — persona-driven, rule-isolated).

One LLM call per persona produces N session skeletons for the session pool.
Each skeleton carries only `{domain, theme_one_line}`. Pool-level scenario
architecture lives here; all session content (instruction, task_params,
references, expected_tools) is filled by the downstream learning-session fill stage.

Design philosophy:
  - The skeleton stage is **persona-driven only**. It does NOT see rule content,
    signal tool hints, or coverage mode.
  - Situational color (time/place/mood) is baked into the theme text rather
    than emitted as structured fields — learning-session fill reads the theme
    and grounds the fill realistically.
  - expected_tools is owned by learning-session fill: tool focus is decided
    alongside the content fill actually writes, eliminating the
    skeleton↔fill mismatch class where the skeleton commits to a tool set
    the fill content can't support.
  - Coverage / difficulty control lives in the later (post-datagen) episode
    composition step.

**Operational note — run AFTER the test-session stage**:
    The candidate pool size is `round(test_sessions_count × 3) + buffer` (see
    `_resolve_n` below). It prefers `test_sessions/` count because those are
    the rules that survived test-session stage QC and will actually be evaluated;
    rules dropped at the test-session stage don't need LS signal. Falling back
    to rules.json (when test_sessions/ is missing) over-counts by ~50-100%,
    wasting learning-session fill LLM calls on skeletons that can never become
    signal. The final episode trajectory is shorter: episode compose samples
    K = test_sessions_count × 2 from this generated pool.

    Recommended order: the rule stage (rules) → the test-session stage
    (test_session_gen) → the learning-session stage (this) → episode compose.
    Running the learning-session stage before the test-session stage will use
    rules.json fallback and produce too many skeletons.

Input:  data/personas/<uuid>/structured.json (+ test_sessions/ for sizing)
Output: data/personas/<uuid>/skeleton.json (the session pool before learning-session fill)

CLI:
    uv run python -m datagen.learning_sessions.skeleton.gen \\
        --persona-id <uuid> [--n 40]
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datagen._common import (
    DOMAINS,
    PERSONAS_DIR,
    archive_to_prev,
    fill_prompt,
    format_structured_persona,
    load_prompt,
    read_json,
    render_all_domain_tools_brief,
    write_json,
)
from datagen.config import CONFIG
from lib.llm import GPT, call_llm_json, model_scope  # type: ignore
from lib.tokens import make_bucket, record_stage  # type: ignore

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
DEFAULT_MODEL = GPT

# ---------------------------------------------------------------------------
# Pool size sizing
# ---------------------------------------------------------------------------


def _rule_count_for_sizing(persona_id: str) -> int:
    """Rule count used to size the LS pool.

    Prefer test_sessions/ count — those are the rules that survived
    test-session stage QC and will actually be evaluated. signal_ls during
    episode_assemble is keyed off test_session.required_actions, so over-sizing
    for rules that were dropped at the test-session stage wastes learning-session
    fill LLM calls (the extra skeletons can never contribute to signal).

    Fallback chain: test_sessions/ → rules.json → CONFIG default.
    """
    pdir = PERSONAS_DIR / persona_id
    ts_dir = pdir / "test_sessions"
    if ts_dir.exists():
        n = sum(1 for _ in ts_dir.glob("*.json"))
        if n > 0:
            return n
    rules_path = pdir / "rules.json"
    if rules_path.exists():
        try:
            return len(read_json(rules_path))
        except Exception:
            pass
    return CONFIG.rule_count_per_persona


def _resolve_n(persona_id: str, cli_override: int | None) -> int:
    """Pick session-pool size N for one persona.

      1. --n from CLI wins
      2. otherwise N = clamp(round(rule_count × multiplier) + buffer,
                             [min, max])
         where rule_count comes from `_rule_count_for_sizing`.
    """
    if cli_override is not None:
        return cli_override
    rule_count = _rule_count_for_sizing(persona_id)
    n = round(rule_count * CONFIG.learning_multiplier) + CONFIG.learning_drop_buffer
    return max(CONFIG.learning_session_min,
               min(CONFIG.learning_session_max, n))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _validate_skeleton(sk: dict, idx: int) -> list[str]:
    """Thin validator — only checks the two fields the skeleton stage commits to.

    `situations` / `references_sketch` / `expected_tools` are no longer
    part of the skeleton contract; they're either owned by learning-session
    fill (expected_tools) or dropped entirely (situations / references_sketch —
    learning-session fill bakes situational color into the instruction text directly).
    """
    errors: list[str] = []
    for field in ("domain", "theme_one_line"):
        v = sk.get(field)
        if v is None or (isinstance(v, str) and not v.strip()):
            errors.append(f"skeleton_{idx}_missing:{field}")
    if errors:
        return errors

    domain = sk["domain"]
    if domain not in DOMAINS:
        errors.append(f"skeleton_{idx}_invalid_domain:{domain}")

    return errors


def _stamp(skeletons: list[dict], persona_id: str) -> list[dict]:
    """Assign sequential session_id + day_offset. For MVP, day_offset is a
    pure index — no timeline semantics.
    """
    out = []
    for i, sk in enumerate(skeletons):
        sk = dict(sk)
        sk["session_id"] = f"{persona_id}_s{i:02d}"
        sk["day_offset"] = i
        out.append(sk)
    return out


def _coverage_report(skeletons: list[dict]) -> dict:
    """Domain coverage only — situations are no longer structured fields."""
    return {"domains": sorted({sk["domain"] for sk in skeletons})}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def generate_skeletons(
    persona_id: str,
    structured: dict,
    n: int,
    model: str,
) -> tuple[list[dict], list[dict]]:
    """Produce N persona-driven skeletons. Returns (passed_stamped, rejected).

    The skeleton stage sees only persona + domain tool briefs — no rule content,
    no signal hints, no coverage mode. Tool choice is natural.
    """
    template = load_prompt(PROMPTS_DIR, name="gen")
    prompt = fill_prompt(
        template,
        STRUCTURED_PERSONA=format_structured_persona(structured),
        N=str(n),
        TOOLS_BRIEF=render_all_domain_tools_brief(),
    )

    logger.info("[%s] calling LLM (n=%d)", persona_id, n)
    t0 = time.time()
    result = call_llm_json(prompt, model=model, temperature=0.7, max_retries=3)
    elapsed = time.time() - t0
    logger.info("[%s] LLM returned in %.1fs (%s items)",
                persona_id, elapsed,
                len(result) if isinstance(result, list) else "non-list")

    if not isinstance(result, list):
        return [], []

    passed: list[dict] = []
    rejected: list[dict] = []
    for i, sk in enumerate(result):
        if not isinstance(sk, dict):
            rejected.append({"_reject_reasons": [f"item_{i}_not_dict"], "raw": sk})
            continue
        reasons = _validate_skeleton(sk, i)
        if reasons:
            sk = dict(sk)
            sk["_reject_reasons"] = reasons
            rejected.append(sk)
        else:
            passed.append(sk)

    passed = _stamp(passed, persona_id)
    return passed, rejected


def run_persona(
    persona_id: str,
    n_override: int | None,
    model: str,
    force: bool,
) -> dict:
    pdir = PERSONAS_DIR / persona_id
    out_path = pdir / "skeleton.json"
    if out_path.exists() and not force:
        logger.info("[%s] skeleton.json exists, skip (use --force)", persona_id)
        return {"persona_id": persona_id, "status": "skipped"}

    structured_path = pdir / "structured.json"
    if not structured_path.exists():
        logger.error("[%s] missing structured.json — run ingest first", persona_id)
        return {"persona_id": persona_id, "status": "missing_structured"}

    # --force protocol: archive previous outputs to *_prev/ slots before
    # regen, so we can compare rounds and recover. Skeleton change
    # invalidates all downstream fills, so learning_sessions/ + tied
    # failure logs are also archived (their content is stale once the
    # skeleton's session_ids re-map to new themes).
    if force:
        archived: list[str] = []
        for p in (
            out_path,                                    # skeleton.json
            pdir / "skeleton_rejected.json",
            pdir / "learning_sessions",
            pdir / "fill_failures.json",
        ):
            bak = archive_to_prev(p)
            if bak is not None:
                archived.append(f"{p.name}→{bak.name}")
        if archived:
            logger.info("[%s] archived to *_prev: %s", persona_id, ", ".join(archived))

    structured = read_json(structured_path)
    n = _resolve_n(persona_id, n_override)
    bucket = make_bucket()
    with model_scope(bucket):
        passed, rejected = generate_skeletons(persona_id, structured, n, model)
    record_stage(pdir, "learning_session", bucket)

    if passed:
        write_json(out_path, passed)
    if rejected:
        write_json(pdir / "skeleton_rejected.json", rejected)

    cov = _coverage_report(passed) if passed else {}
    logger.info("[%s] %d passed, %d rejected | domain cov: %s",
                persona_id, len(passed), len(rejected), cov)
    return {
        "persona_id": persona_id,
        "status": "ok",
        "passed": len(passed),
        "rejected": len(rejected),
        "coverage": cov,
    }


def main():
    parser = argparse.ArgumentParser(description="Learning-session skeleton (pool mode)")
    parser.add_argument("--persona-id", type=str, default=None,
                        help="prefix match; if omitted, process all personas with structured.json")
    parser.add_argument(
        "--n", type=int, default=None,
        help=(
            f"override pool size per persona; default derives from rules "
            f"(rule_count × {CONFIG.learning_multiplier}, clamped to "
            f"[{CONFIG.learning_session_min}, {CONFIG.learning_session_max}])"
        ),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
        "processing %d personas, model=%s (concurrency capped by llm.PER_MODEL_MAX_CONCURRENCY)",
        len(all_pids), args.model,
    )
    with ThreadPoolExecutor(max_workers=max(1, len(all_pids))) as pool:
        futs = {
            pool.submit(run_persona, pid, args.n, args.model, args.force): pid
            for pid in all_pids
        }
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                logger.info("[%s] %s", pid, fut.result())
            except Exception as e:
                logger.error("[%s] failed: %s", pid, e, exc_info=True)


if __name__ == "__main__":
    main()
