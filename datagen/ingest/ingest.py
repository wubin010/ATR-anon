"""Ingest Nemotron parquet records into the ATRBench persona layout.

Two-tier source:
  1. `data/persona_selection_formal20.json` — **selector only** by default,
     contributes uuid set for the formal 20-persona cohort. A different
     selector file can be passed with `--selection`.
     (+ optional layer filter). The selection's narrative / slug / demographics
     are NOT used; they were a derived snapshot and lossy.
  2. `Nemotron-Personas-USA/data/train-*.parquet` — **single source of truth**
     for full per-persona data (demographics, 5 narrative subfields,
     skills_and_expertise_list, hobbies_and_interests_list,
     career_goals_and_ambitions, bachelors_field, zipcode, etc.).

Per persona:
  - One LLM call: extract `name` + compress 9 narrative subfields into a
    ≤500-word `complementary_info` paragraph. `raw_persona.narrative` becomes
    user_sim's runtime background card.
  - persona_id slug derived from extracted name (Unicode NFKD normalize +
    snake_case). Collisions get `_2` / `_3` suffix via atomic mkdir.
  - re-runs with `--force`: matched by `raw.original_uuid` → reuse existing dir.

raw.json schema:
  - `uuid`           = persona_id slug (runtime identity)
  - `original_uuid`  = Nemotron 32-hex uuid, hyphens stripped
  - `narrative`      = formatted card (user-sim's background)
  - `general_domain` / `specific_domain` = occupation / bachelors_field

Output: data/personas/<slug>/
  - raw.json
  - structured.json (full Nemotron record reorganized for rule_gen / session_gen)

CLI:
    uv run python -m datagen.ingest.ingest [--layer 20] \\
        [--persona-id <slug_or_uuid_prefix>] [--force]
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datagen._common import PERSONAS_DIR, PROJECT_ROOT, read_json, write_json
from lib.llm import GPT, call_llm_json, model_scope  # type: ignore
from lib.tokens import make_bucket, record_stage  # type: ignore

DEFAULT_SELECTION = PROJECT_ROOT / "data" / "persona_selection_formal20.json"
NEMOTRON_PARQUET_DIR = PROJECT_ROOT / "Nemotron-Personas-USA" / "data"
DEFAULT_MODEL = GPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_list(value: Any) -> list[str]:
    """Nemotron parquet stores `*_list` fields as Python-repr strings like
    "['Bridge (card game)', 'Vegetable gardening']" (str column with list
    syntax inside). Use literal_eval — single quotes aren't JSON-valid.
    """
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass
    # fallback: comma-split
    return [x.strip() for x in s.split(",") if x.strip()]


def _basic_profile(persona: dict) -> dict:
    education = (persona.get("education_level") or "").strip()
    bachelors = (persona.get("bachelors_field") or "").strip()
    if bachelors:
        education = f"{education} ({bachelors.replace('_', ' ')})".strip()

    hobbies = _coerce_list(persona.get("hobbies_and_interests_list"))
    skills = _coerce_list(persona.get("skills_and_expertise_list"))

    return {
        "age": persona.get("age"),
        "sex": persona.get("sex", ""),
        "marital_status": persona.get("marital_status", ""),
        "education_level": education or "unspecified",
        "occupation": persona.get("occupation", ""),
        "city": persona.get("city", ""),
        "state": persona.get("state", ""),
        "zipcode": persona.get("zipcode", ""),
        "country": persona.get("country", "USA"),
        "hobbies_and_interests": ", ".join(hobbies) if hobbies else "N/A",
        "skills_and_expertise": ", ".join(skills) if skills else "N/A",
    }


def _build_prompt(basic: dict, complementary: dict) -> str:
    basic_lines = "\n".join(
        f"{k}: {basic.get(k)}"
        for k in ("age", "sex", "marital_status", "education_level", "occupation",
                  "city", "state", "hobbies_and_interests", "skills_and_expertise")
    )
    comp_json = json.dumps(complementary, indent=2, ensure_ascii=False)
    return (
        "You have two tasks:\n"
        "1. Extract the person's full name from the complementary information.\n"
        "2. Write a concise paragraph (≤ 500 words, English) summarizing the "
        "complementary information. Include ONLY details that cannot be "
        "derived from the basic profile. Focus on lived routines, recurring "
        "contexts, named places/people, cultural background, and long-term "
        "life patterns that would help a role-play LLM stay in character as "
        "this user. Do NOT editorialize their preferences — stay descriptive.\n\n"
        f"Basic Profile:\n{basic_lines}\n\n"
        f"Complementary Information:\n{comp_json}\n\n"
        "Respond in strict JSON with `name` and `profile` keys."
    )


def llm_extract_name_and_summary(persona: dict, model: str) -> tuple[str, str]:
    """Format a Nemotron persona into (name, complementary_info_summary)
    via the persona-profile formatter used at sampling time.
    """
    basic = _basic_profile(persona)
    complementary = {
        k: persona.get(k, "")
        for k in ("persona", "professional_persona", "sports_persona",
                  "arts_persona", "travel_persona", "culinary_persona",
                  "career_goals_and_ambitions",
                  "cultural_background", "skills_and_expertise",
                  "hobbies_and_interests")
        if persona.get(k)
    }
    prompt = _build_prompt(basic, complementary)
    result = call_llm_json(prompt, model=model, temperature=0.3, max_retries=3)
    if not isinstance(result, dict):
        raise RuntimeError(f"LLM returned non-dict: {type(result).__name__}")
    return (result.get("name") or "N/A").strip(), (result.get("profile") or "").strip()


def build_formatted_narrative(basic: dict, name: str, complementary: str) -> str:
    """Render the user-sim background card. Ordered so a runtime user_sim
    reads identity first, then basic facts, then narrative context."""
    lines = [
        f"name: {name}",
        f"age: {basic.get('age')}",
        f"sex: {basic.get('sex')}",
        f"marital_status: {basic.get('marital_status')}",
        f"education_level: {basic.get('education_level')}",
        f"occupation: {basic.get('occupation')}",
        f"hobbies_and_interests: {basic.get('hobbies_and_interests')}",
        f"skills_and_expertise: {basic.get('skills_and_expertise')}",
        f"complementary_info: {complementary}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shape → raw / structured / persona_profile
# ---------------------------------------------------------------------------


def shape_raw(
    persona: dict, narrative: str, persona_id: str, original_uuid: str
) -> dict:
    """RawPersona shape consumed by runtime. `narrative` is the user-sim card.

    `persona_id` is the slug (e.g. "jutta_ashton") — runtime treats this as
    the identity key. `nemotron_uuid` is Nemotron's 32-hex uuid, kept for
    dedup on re-runs and data provenance.
    """
    return {
        "persona_id": persona_id,
        "nemotron_uuid": original_uuid,
        "narrative": narrative,
        "general_domain": (persona.get("occupation") or "").strip() or "",
        "specific_domain": (persona.get("bachelors_field") or "").strip() or "",
    }


def shape_structured(persona: dict, name: str) -> dict:
    """Full rich record for rule_gen / session_gen to consume.
    Preserves Nemotron's narrative subfields for downstream LLM prompts.
    """
    basic = _basic_profile(persona)
    hobbies = _coerce_list(persona.get("hobbies_and_interests_list"))
    skills = _coerce_list(persona.get("skills_and_expertise_list"))

    return {
        "name": name,
        "demographics": {
            "age": basic["age"],
            "sex": basic["sex"],
            "marital_status": basic["marital_status"],
            "education_level": basic["education_level"],
            "occupation": basic["occupation"],
            "city": basic["city"],
            "state": basic["state"],
            "zipcode": basic["zipcode"],
            "country": basic["country"],
        },
        "persona_summary": (persona.get("persona") or "").strip(),
        "professional_persona": (persona.get("professional_persona") or "").strip(),
        "sports_persona": (persona.get("sports_persona") or "").strip(),
        "arts_persona": (persona.get("arts_persona") or "").strip(),
        "travel_persona": (persona.get("travel_persona") or "").strip(),
        "culinary_persona": (persona.get("culinary_persona") or "").strip(),
        "cultural_background": (persona.get("cultural_background") or "").strip(),
        "skills_and_expertise": (persona.get("skills_and_expertise") or "").strip(),
        "skills_and_expertise_list": skills,
        "hobbies_and_interests": (persona.get("hobbies_and_interests") or "").strip(),
        "hobbies_and_interests_list": hobbies,
        "career_goals_and_ambitions": (persona.get("career_goals_and_ambitions") or "").strip(),
        # selection_reason is dropped at the boundary — see module docstring.
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _original_uuid(rec: dict) -> str:
    """Nemotron's uuid field, stripped of hyphens. Falls back to sha256 hash
    of persona text when the record somehow lacks a uuid."""
    u = (rec.get("uuid") or "").strip().replace("-", "")
    if u:
        return u
    from hashlib import sha256
    return sha256((rec.get("persona") or "").encode("utf-8")).hexdigest()


def _slug_from_name(name: str) -> str | None:
    """Convert an extracted name ("Jutta Ashton" / "Juan Marrero") into a
    filesystem-safe slug. Returns None if the name is empty / "N/A" / produces
    an empty slug after normalization.

    - Unicode NFKD normalize → strip combining chars (Juán → Juan)
    - lowercase
    - non-[a-z0-9] runs → single underscore
    - strip leading/trailing underscores
    """
    if not name:
        return None
    s = name.strip()
    if not s or s.upper() in ("N/A", "NONE", "UNKNOWN"):
        return None
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = s.strip("_")
    return s or None


def _find_existing_slug(original_uuid: str) -> str | None:
    """Scan PERSONAS_DIR for a directory whose raw.json.nemotron_uuid matches.
    Used to reuse an existing slug on re-runs (--force) without creating
    a collision-suffixed duplicate.
    """
    if not PERSONAS_DIR.is_dir():
        return None
    for d in PERSONAS_DIR.iterdir():
        if not d.is_dir():
            continue
        raw_path = d / "raw.json"
        if not raw_path.exists():
            continue
        try:
            data = read_json(raw_path)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        # Also accept the "original_uuid" key
        nuuid = data.get("nemotron_uuid") or data.get("original_uuid") or ""
        if nuuid == original_uuid:
            return d.name
    return None


def _reserve_slug_dir(base: str) -> str:
    """Atomically pick a unique slug by trying `base`, `base_2`, `base_3`, ...
    via `mkdir(exist_ok=False)`. Returns the slug that was successfully
    mkdir'd — caller is responsible for writing files into it.
    """
    slug = base
    i = 1
    while True:
        try:
            (PERSONAS_DIR / slug).mkdir(parents=True, exist_ok=False)
            return slug
        except FileExistsError:
            i += 1
            slug = f"{base}_{i}"


def assign_persona_id(name: str, original_uuid: str) -> tuple[str, bool]:
    """Pick the persona_id slug for this record. Returns (slug, reused).

    Logic:
    1. If an existing dir already has raw.original_uuid == ours, reuse that
       slug (force re-run case) — `reused=True`.
    2. Otherwise derive base from the LLM-extracted name (slugified), or fall
       back to original_uuid[:12] if the name can't be slugified.
    3. Atomically reserve a unique slug dir (add _2 / _3 suffix on collision).
    """
    existing = _find_existing_slug(original_uuid)
    if existing is not None:
        return existing, True
    base = _slug_from_name(name) or original_uuid[:12]
    slug = _reserve_slug_dir(base)
    return slug, False


def has_output(persona_id: str) -> bool:
    d = PERSONAS_DIR / persona_id
    return (d / "structured.json").exists()


def has_output_for_record(rec: dict) -> bool:
    """Check if this Nemotron record already has a persona dir (matched by
    original_uuid). Used in main() to skip already-processed records before
    we spend an LLM call extracting the name for slug generation.
    """
    return _find_existing_slug(_original_uuid(rec)) is not None


def process_one(rec: dict, model: str) -> tuple[str, dict, dict, dict]:
    original_uuid = _original_uuid(rec)

    t0 = time.time()
    bucket = make_bucket()
    with model_scope(bucket):
        name, complementary = llm_extract_name_and_summary(rec, model)
    persona_id, reused = assign_persona_id(name, original_uuid)
    basic = _basic_profile(rec)
    narrative = build_formatted_narrative(basic, name, complementary)
    logger.info(
        "[%s] extracted name=%r narrative=%d chars in %.1fs (reused_dir=%s)",
        persona_id, name, len(narrative), time.time() - t0, reused,
    )

    raw = shape_raw(rec, narrative, persona_id, original_uuid)
    structured = shape_structured(rec, name)
    return persona_id, raw, structured, bucket


def save_one(persona_id: str, raw: dict, structured: dict) -> Path:
    pdir = PERSONAS_DIR / persona_id
    pdir.mkdir(parents=True, exist_ok=True)  # already created by assign_persona_id
    write_json(pdir / "raw.json", raw)
    write_json(pdir / "structured.json", structured)
    return pdir


def _norm_uuid(u: str) -> str:
    """Drop hyphens, lowercase. Selection.json uses no-hyphen 32-hex;
    parquet uses standard UUID with hyphens. Match on the normalized form.
    """
    return (u or "").replace("-", "").lower()


def load_uuids_from_selection(
    path: Path, layer: int | None = None,
) -> set[str]:
    """Read the persona selection JSON and return the set of uuid (no-hyphen,
    lowercase) for the requested layer subset. Layer semantics per
    `meta.nesting`: L5 ⊂ L10 ⊂ L20 ⊂ L40 ⊂ L100. `--layer N` includes
    every record with `entry.layer <= N`. layer=None → all 100 uuids.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    personas = data.get("personas")
    if not isinstance(personas, list):
        raise ValueError(f"{path}: missing or invalid 'personas' field")
    out: set[str] = set()
    for p in personas:
        if layer is not None and (p.get("layer") or 9999) > layer:
            continue
        u = _norm_uuid(p.get("uuid", ""))
        if u:
            out.add(u)
    return out


def load_from_parquet(uuids: set[str]) -> list[dict]:
    """Fetch full Nemotron records by uuid. Uses pyarrow parquet
    push-down filter so only matching row groups get decoded.

    Both selection.json and parquet store uuid as 32-hex no-hyphen, so
    no format conversion is needed.

    Returns a list of dicts, each carrying the full Nemotron schema
    (skills_and_expertise_list / hobbies_and_interests_list /
    career_goals_and_ambitions / bachelors_field / zipcode /
    professional_persona etc.).
    """
    import glob
    import pandas as pd

    if not uuids:
        return []

    targets = list(uuids)
    out_by_uuid: dict[str, dict] = {}
    parquet_files = sorted(glob.glob(str(NEMOTRON_PARQUET_DIR / "train-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"no parquet files under {NEMOTRON_PARQUET_DIR} — clone "
            "Nemotron-Personas-USA first"
        )

    for fp in parquet_files:
        df = pd.read_parquet(fp, filters=[("uuid", "in", targets)])
        if df.empty:
            continue
        for rec in df.to_dict(orient="records"):
            key = _norm_uuid(rec.get("uuid", ""))
            if key in uuids and key not in out_by_uuid:
                out_by_uuid[key] = rec
        if len(out_by_uuid) == len(uuids):
            break  # all uuids resolved, stop scanning

    missing = uuids - set(out_by_uuid)
    if missing:
        logger.warning(
            "%d uuid(s) not found in any parquet: %s",
            len(missing), sorted(missing)[:5],
        )

    return list(out_by_uuid.values())


def main():
    parser = argparse.ArgumentParser(
        description="Ingest: ingest from Nemotron parquet (selection.json = uuid selector only)"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="persona selection JSON (uuid selector; default: data/persona_selection_formal20.json)",
    )
    parser.add_argument("--layer", type=int, default=None,
                        choices=[5, 10, 20, 40, 100],
                        help="filter to layered subset (L5⊂L10⊂L20⊂L40⊂L100)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--persona-id", type=str, default=None,
                        help="prefix match against uuid[:12] OR existing dir slug")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.selection.exists():
        parser.error(f"selection not found: {args.selection}")

    target_uuids = load_uuids_from_selection(args.selection, layer=args.layer)
    logger.info(
        "selection: %d uuids%s",
        len(target_uuids),
        f" (layer<={args.layer})" if args.layer else " (all layers)",
    )

    records = load_from_parquet(target_uuids)
    logger.info(
        "fetched %d records from Nemotron parquet under %s",
        len(records), NEMOTRON_PARQUET_DIR,
    )

    if args.persona_id:
        # Filter matches: uuid[:12] prefix OR existing dir slug prefix
        def _matches(r: dict) -> bool:
            if _original_uuid(r).startswith(args.persona_id):
                return True
            existing = _find_existing_slug(_original_uuid(r))
            return bool(existing and existing.startswith(args.persona_id))
        records = [r for r in records if _matches(r)]
        if not records:
            parser.error(f"no persona matched --persona-id {args.persona_id!r}")
    if args.limit:
        records = records[: args.limit]

    if not args.force:
        before = len(records)
        records = [r for r in records if not has_output_for_record(r)]
        skipped = before - len(records)
        if skipped:
            logger.info("skipped %d already-processed (use --force to overwrite)", skipped)

    if not records:
        logger.info("nothing to do.")
        return

    logger.info("processing %d personas, model=%s (concurrency capped by llm.PER_MODEL_MAX_CONCURRENCY)",
                len(records), args.model)
    stats = {"ok": 0, "fail": 0}

    with ThreadPoolExecutor(max_workers=max(1, len(records))) as pool:
        futs = {pool.submit(process_one, r, args.model): r for r in records}
        for fut in as_completed(futs):
            r = futs[fut]
            orig = _original_uuid(r)[:12]
            try:
                persona_id, raw, structured, bucket = fut.result()
                pdir = save_one(persona_id, raw, structured)
                record_stage(pdir, "persona_preprocess", bucket)
                stats["ok"] += 1
                city = structured.get("demographics", {}).get("city", "")
                state = structured.get("demographics", {}).get("state", "")
                logger.info("[%s] saved (%s, %s, %s)", persona_id,
                            structured.get("name", "?"), city, state)
            except Exception as e:
                stats["fail"] += 1
                logger.error("[%s…] failed: %s", orig, e, exc_info=True)

    logger.info("done. ok=%d fail=%d", stats["ok"], stats["fail"])


if __name__ == "__main__":
    main()
