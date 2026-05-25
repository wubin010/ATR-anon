"""Learning-session fill: per-skeleton LearningSession fill.

One LLM call per skeleton — parallel across workers. Reads skeleton.json
(thin: {domain, theme_one_line, session_id, day_offset}) + structured.json,
produces one complete LearningSession per skeleton written to
`data/personas/<uuid>/learning_sessions/<session_id>.json`.

LLM-produced fields:
  - instruction, task_params, expected_tools, local_env.references

Programmatic additions (not from LLM):
  - session_id, session_type, day_offset, domain  — copied from skeleton
  - local_env.tools                               — set to whole-domain tool list

Input:  data/personas/<uuid>/skeleton.json + structured.json
Output: data/personas/<uuid>/learning_sessions/<session_id>.json (per-skeleton)

CLI:
    uv run python -m datagen.learning_sessions.fill.gen --persona-id <uuid>
        [--session-id <sid>] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datagen._common import (
    DOMAINS,
    PERSONAS_DIR,
    archive_to_prev,
    base_tool_names,
    domain_tools,
    fill_prompt,
    format_structured_persona,
    load_prompt,
    read_json,
    ref_types_for_domain,
    render_domain_tools,
    render_ref_schema,
    to_json,
    tool_map,
    write_json,
)
from lib.llm import GPT, call_llm_json, model_scope  # type: ignore
from lib.tokens import make_bucket, record_stage  # type: ignore
from runner.schemas import LearningSession  # schema-aware retry validation

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"
DEFAULT_MODEL = GPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference types allowed per domain — dynamically resolved from TableSpec source of truth.
# See datagen._common.ref_types_for_domain().

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Heuristics for quality floor #2: fields whose NAME hints at "output text"
# that the agent should author (not the user). If value is a long string in
# one of these fields, LLM is pre-authoring agent work — reject.
_OUTPUT_LIKE_FIELD_TOKENS = (
    "body", "content", "reply", "draft", "text",
    "email_body", "message_body", "document_content",
)
_OUTPUT_VALUE_MAX_CHARS = 200

# task_params field count band (quality floor #3). Prompt biases LLM toward
# 3-10, but the validator tolerates up to 12 to avoid dropping sessions over
# a single over-field. Still reject ≥13 (likely structural mistake).
_TASK_PARAMS_MIN = 3
_TASK_PARAMS_MAX = 12

# Runtime-autofilled identity fields — allowed to appear as id-ish values
# without a matching reference object (env injects them from PersonaProfile).
_RUNTIME_AUTOFILLED = {
    "contact_info", "traveler_info", "guest_info",
    "shipping_address", "payment_method",
}

_WRITE_ARG_GROUNDING_OPTIONAL_DEFAULTS = {
    ("draft_message", "language"),
    ("send_message", "language"),
}

_WRITE_ARG_GROUNDING_CONTROL_ARGS = {
    "field", "update_action",
    "include_daily_breakdown", "include_stopovers", "include_budget_summary",
}

_WRITE_ARG_GROUNDING_ID_ARGS = {
    "thread_id", "target_id", "event_id", "message_id", "message_ids",
    "file_id", "file_ids", "folder_id", "label_id", "label_ids",
    "document_id", "tracker_id", "entry_id", "booking_id",
    "reservation_id", "appointment_id", "provider_id", "product_id",
    "product_ids", "selected_stop_ids", "destination_ids", "flight_id",
    "hotel_offer_id", "ground_offer_id", "restaurant_id",
    "service_provider_id", "trip_id",
}

_WRITE_ARG_GROUNDING_WRITE_TOOLS = {
    "place_order", "modify_order", "cancel_order", "return_order",
    "book_restaurant", "modify_restaurant_reservation",
    "cancel_restaurant_reservation", "book_service_appointment",
    "modify_service_appointment", "cancel_service_appointment",
    "book_event_ticket", "modify_event_ticket", "cancel_event_ticket",
    "book_flight", "book_hotel", "book_ground_transport", "plan_trip",
    "replan_trip", "draft_message", "send_message", "send_draft",
    "set_message_priority", "archive_messages", "label_messages",
    "create_document", "update_document", "move_files", "archive_files",
    "delete_files", "classify_files", "update_tracker", "create_event",
    "reschedule_event", "modify_event", "respond_to_event_invite",
    "cancel_event", "set_reminder", "track_event_updates",
}

_SEARCH_ARG_GROUNDING_OPTIONAL_FILTERS = {
    # Optional narrowing. Omitting these filters returns a recoverable broad
    # list, and current user_sim may not disclose these buckets naturally.
    ("list_events", "calendar"),
}

_SEARCH_ARG_GROUNDING_REQUIRED_ARGS = {
    "search_products": {"category", "budget"},
    "search_restaurants": {"location", "cuisine"},
    "search_service_providers": {"service_type", "location"},
    "search_events": {"location", "event_tags"},
    "search_destinations": {"destination_constraints"},
    "search_trip_stops": {"destination", "stop_tags"},
    "search_flights": {"origin", "destination", "passenger_count"},
    "search_hotels": {"location", "guest_count", "room_count"},
    "search_ground_transport": {"origin", "destination", "passenger_count"},
    "list_files": {"location", "file_type"},
    "search_documents": {"location", "doc_type"},
    "search_messages": {"sender", "folder"},
    "list_events": {"calendar", "participants"},
}

_SEARCH_DOCUMENT_SOURCE_DOC_TYPE_KEYS = {
    "source_doc_type",
    "template_doc_type",
    "reference_doc_type",
}

# task_param keys whose corresponding objects are RESULT objects of past
# tool calls (orders placed, trips planned, bookings made, appointments
# created), not pre-existing candidate objects. The current ref schema
# intentionally has no ref type for these (commerce: no `order`; travel:
# no `trip` / `booking`; reservation: no `appointment`) — they must be
# rewritten as `<noun>_description` fields so the agent fetches them via
# track_*/search_* at runtime. The validator emits a more directive error
# when one of these keys appears, since the generic "add to refs" hint
# misleads the LLM into the (forbidden) refs path.
_RESULT_OBJECT_ID_KEYS = {
    "order_id", "trip_id", "booking_id", "appointment_id",
    "reservation_id",
}

# Env-preseeded reference ids — runtime injects these into local_env
# regardless of what the session declares. Currently empty (priority migrated
# from auto-preseeded buckets to a flat enum on set_message_priority.priority,
# so no auto-injected ref ids). Mirrors `_AUTO_PRESEED_IDS` in
# test_session_gen/static_check.py — keep in sync.
_AUTO_PRESEED_IDS: set[str] = set()

# Known reference-id prefixes (mirror of REF_TYPE_SCHEMA.md id conventions,
# plus a few informal task-param-side ids like `ord_*` for orders / `thr_*`
# for threads that the datagen prompt allows in desc-only shape but that sometimes
# leak into values). Used to spot hard-coded ids in field values when the field
# name doesn't end in `_id` / `_ids`. Snake-case enum values like
# `casual_american` or `by_priority` won't match any of these prefixes.
_REF_ID_PREFIXES: tuple[str, ...] = (
    "rst_", "prod_", "sub_", "evt_", "prv_",
    "dst_", "flt_", "hot_", "gnd_", "stp_",
    "msg_", "pb_", "fld_", "lbl_",
    "fil_", "doc_", "trk_",
    "ord_", "thr_", "bkg_", "bk_",
)

# Search-style tools → ref types they target. Built dynamically from
# TableSpec declarations (single source of truth).


def _build_search_tool_ref_types() -> dict[str, set[str]]:
    from runner.environment.tables import all_table_specs
    out: dict[str, set[str]] = {}
    for _, spec in all_table_specs():
        if spec.kind != "primary":
            continue
        ref_types = {spec.source_ref_type, *(spec.aliases or [])} - {None}
        for tool in spec.discovery_tools:
            out.setdefault(tool, set()).update(ref_types)
    return out


_SEARCH_TOOL_REF_TYPES: dict[str, set[str]] = _build_search_tool_ref_types()


def _looks_like_id_value(v: Any) -> bool:
    """String value with a known ref-id prefix AND at least one digit.

    Requiring a digit prevents false positives on normal words that happen
    to share a short prefix (e.g. "hot_spring" matching the `hot_` hotel
    prefix — real ids always carry a numeric component like `hot_marriott_02`).
    """
    if not isinstance(v, str) or not v.startswith(_REF_ID_PREFIXES):
        return False
    return any(c.isdigit() for c in v)


def _extract_id_values(key: str, value: Any) -> list[str]:
    """Return the id-ish strings inside a task_param value, if any.

    Triggers:
      - field name ends in `_id` → whole value counts as an id (even if the
        string doesn't carry a known prefix, since the field name itself is
        a strong signal)
      - field name ends in `_ids` → every list item counts as an id
      - otherwise scalar string with known ref-id prefix (e.g. `ord_art_48219`)
      - list of strings where any item has known prefix
    """
    out: list[str] = []
    if key.endswith("_id") and isinstance(value, str):
        out.append(value)
    elif key.endswith("_ids") and isinstance(value, list):
        for v in value:
            if isinstance(v, str):
                out.append(v)
    elif isinstance(value, str) and _looks_like_id_value(value):
        out.append(value)
    elif isinstance(value, list):
        for v in value:
            if _looks_like_id_value(v):
                out.append(v)
    return out


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        out: list[Any] = []
        for item in value.values():
            out.extend(_flatten_values(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [value]


def _norm_grounded_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _task_param_value_pool(tp: dict) -> set[str]:
    out: set[str] = set()
    for value in (tp or {}).values():
        out.add(_norm_grounded_value(value))
        for atomic in _flatten_values(value):
            out.add(_norm_grounded_value(atomic))
    return out


def _is_write_id_arg(arg_name: str) -> bool:
    return (
        arg_name in _WRITE_ARG_GROUNDING_ID_ARGS
        or arg_name.endswith("_id")
        or arg_name.endswith("_ids")
    )


def _is_authored_placeholder(value: Any, tp: dict) -> bool:
    if not isinstance(value, str):
        return False
    m = re.fullmatch(r"<authored from (.+)>", value.strip())
    if not m:
        return False
    fields = [
        part.strip()
        for part in re.split(r",|\band\b", m.group(1))
        if part.strip()
    ]
    return bool(fields) and all(field in tp for field in fields)


def _is_empty_filter_value(value: Any) -> bool:
    return value is None or value == "" or value == []


def _is_recoverable_optional_search_filter(
    tool_name: str,
    arg_name: str,
    value: Any,
) -> bool:
    if (tool_name, arg_name) in _SEARCH_ARG_GROUNDING_OPTIONAL_FILTERS:
        return True
    if tool_name == "search_messages" and arg_name == "folder":
        # search_messages() without folder returns all non-archived messages,
        # so inbox/sent/drafts can still be recovered by broad search. Archive
        # is different: archived messages are hidden unless folder="archive".
        return _norm_grounded_value(value) != "archive"
    return False


def _is_value_grounded_in_task_params(value: Any, tp: dict) -> bool:
    if _is_empty_filter_value(value):
        return True

    task_pool = _task_param_value_pool(tp)
    full_value = _norm_grounded_value(value)
    if full_value in task_pool:
        return True

    atoms = [
        _norm_grounded_value(v)
        for v in _flatten_values(value)
        if not _is_empty_filter_value(v)
    ]
    if not atoms:
        return True
    return all(atom in task_pool for atom in atoms)


def _is_document_doc_type_grounded(value: Any, tp: dict) -> bool:
    grounded_keys = {"doc_type", *_SEARCH_DOCUMENT_SOURCE_DOC_TYPE_KEYS}
    grounded_values = {
        _norm_grounded_value(tp[k])
        for k in grounded_keys
        if k in tp
    }
    return _norm_grounded_value(value) in grounded_values


def _validate_search_args_grounded_in_task_params(
    trajectory: list,
    tp: dict,
) -> list[str]:
    """Gold discovery filters must be askable from task_params.

    Dry-run proves a filter works against refs, but it cannot prove the
    simulated user can reveal that filter to the agent. This check catches
    gold-only search anchors such as searching `doc_type="template"` while
    task_params only says the user wants to create a `doc_type="note"`.

    Deliberately excluded: optional broad-list narrowers such as
    list_events.calendar and non-archive search_messages.folder values,
    where omitting the filter remains a normal recoverable path.
    """
    errors: list[str] = []
    for i, step in enumerate(trajectory or []):
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        required_args = _SEARCH_ARG_GROUNDING_REQUIRED_ARGS.get(tool_name)
        if not required_args:
            continue
        args = step.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        for ak, av in args.items():
            if ak not in required_args:
                continue
            if _is_recoverable_optional_search_filter(tool_name, ak, av):
                continue
            if tool_name == "search_documents" and ak == "doc_type":
                if _is_document_doc_type_grounded(av, tp):
                    continue
                errors.append(
                    f"gold_step_{i}_search_doc_type_not_grounded:"
                    f"{tool_name}.{ak}={av!r} "
                    "(search document type is not disclosed by task_params. "
                    "If this is a template/reference search, add "
                    "source_doc_type/template_doc_type/reference_doc_type to "
                    "task_params; otherwise use task_params.doc_type.)"
                )
                continue
            if _is_value_grounded_in_task_params(av, tp):
                continue
            errors.append(
                f"gold_step_{i}_search_arg_not_in_task_params:"
                f"{tool_name}.{ak}={av!r} "
                "(gold discovery filters must be explainable from task_params; "
                "otherwise the simulated user cannot disclose the search anchor)"
            )
    return errors


def _validate_write_args_grounded_in_task_params(
    trajectory: list,
    tp: dict,
    tmap: dict,
) -> list[str]:
    """Concrete user-visible write args must be known to user_sim.

    IDs can be discovered through prior READ tools / refs, and placeholders
    such as `<authored from reply_points>` are grounded in task_params. But a
    concrete write value like create_document.doc_type, create_event.title, or
    set_message_priority.priority must appear in task_params; otherwise the
    gold path asks the agent to write information the simulated user cannot
    disclose.
    """
    task_pool = _task_param_value_pool(tp)
    errors: list[str] = []
    for i, step in enumerate(trajectory or []):
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        if tool_name not in _WRITE_ARG_GROUNDING_WRITE_TOOLS:
            continue
        args = step.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        for ak, av in args.items():
            if _is_write_id_arg(ak):
                continue
            if ak in _WRITE_ARG_GROUNDING_CONTROL_ARGS:
                continue
            if (tool_name, ak) in _WRITE_ARG_GROUNDING_OPTIONAL_DEFAULTS:
                continue
            if _is_authored_placeholder(av, tp):
                continue
            if _is_value_grounded_in_task_params(av, tp):
                continue
            errors.append(
                f"gold_step_{i}_write_arg_not_in_task_params:"
                f"{tool_name}.{ak}={av!r} "
                "(concrete user-visible write args must appear in task_params, "
                "or be rewritten as an <authored from ...> placeholder grounded "
                "in task_params; do not invent write values only in gold_trajectory)"
            )
    return errors


def _is_output_like_field(name: str) -> bool:
    nm = name.lower()
    return any(tok in nm for tok in _OUTPUT_LIKE_FIELD_TOKENS)


def _tokens(s: str) -> set[str]:
    """Token set after splitting on common separators, lowercased.

    Mirrors `_loose_string_match` in runtime env (`runner/environment/base.py`)
    so the validator's "lexical reachability" check approximates what the
    runtime filter will accept.
    """
    import re as _re
    return set(_re.split(r"[\s,./\-_]+", s.lower())) - {""}


def _is_enum_or_id_or_date(s: str) -> bool:
    """Heuristic: skip values that aren't natural search filters.

    - enum literals: snake_case-ish, no spaces, no digits → tool-arg
      destination (A), not a filter against refs
    - ISO dates: matched separately
    - id-shaped (handled by floor #3)
    """
    import re as _re
    if not isinstance(s, str):
        return True
    s = s.strip()
    if not s:
        return True
    # ISO date
    if _re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return True
    # enum literal: `casual_american`, `by_priority`, etc.
    if " " not in s and _re.fullmatch(r"[a-z][a-z0-9_]*", s.lower()):
        return True
    # id-like (floor #3 handles existence)
    if _looks_like_id_value(s):
        return True
    return False


def _validate_autofill(tp: dict) -> list[str]:
    """Reject autofill keys in task_params (cheap shape-level check).

    autofill fields (`contact_info` / `payment_method` / `shipping_address`
    / `traveler_info` / `guest_info`) are runtime-injected from
    PersonaProfile by the env. Appearing in task_params is redundant and
    causes user_sim to disclose values the agent shouldn't be asking
    about anyway.
    """
    errors: list[str] = []
    for k in (tp or {}).keys():
        if k in _RUNTIME_AUTOFILLED:
            errors.append(
                f"task_param_uses_autofill:{k} "
                "(env auto-injects from PersonaProfile; remove from task_params)"
            )
    return errors


def _validate_gold_trajectory(
    trajectory: list, tp: dict, refs: list, domain: str, tmap: dict,
) -> list[str]:
    """Verify the oracle trajectory is internally coherent.

    Subsumes the previous F / B / id-coverage checks: if learning-session fill can write a
    valid trajectory, the LS data must be self-consistent (search-style
    steps reference real ref types; id args resolve to real refs; value
    args appear in task_params or refs).

    Pass criteria (any failure → reject the LS):
      1. trajectory is a non-empty list of {tool, arguments}
      2. each step.tool ∈ this domain's tool set
      3. trajectory length ∈ [1, 5]
      4. every search-style step has ≥1 ref of corresponding type
      5. every id-shaped argument value exists in local_env.references
      6. arguments don't reference autofill fields (env auto-injects;
         keeping them in trajectory shadows that behavior)
    """
    errors: list[str] = []
    if not isinstance(trajectory, list) or not trajectory:
        errors.append("gold_trajectory_empty_or_not_list")
        return errors
    if len(trajectory) > 5:
        errors.append(f"gold_trajectory_too_long:{len(trajectory)}>5")
    domain_tool_names = {t["name"] for t in domain_tools(domain)}
    ref_types_present: set[str] = {
        r["type"] for r in refs
        if isinstance(r, dict) and isinstance(r.get("type"), str)
    }
    # Valid id pool covers BOTH top-level ref.id AND id-shaped strings in
    # any attribute value (e.g. message ref carries `attributes.thread_id`
    # like `thr_safety_01` — that thread_id is a legitimate value for
    # draft_message.thread_id even though it's not a top-level ref.id).
    valid_id_pool: set[str] = set()
    for r in refs:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if isinstance(rid, str):
            valid_id_pool.add(rid)
        attrs = r.get("attributes") or {}
        if isinstance(attrs, dict):
            for av in attrs.values():
                if isinstance(av, str) and _looks_like_id_value(av):
                    valid_id_pool.add(av)
                elif isinstance(av, list):
                    for item in av:
                        if isinstance(item, str) and _looks_like_id_value(item):
                            valid_id_pool.add(item)

    for i, step in enumerate(trajectory):
        if not isinstance(step, dict):
            errors.append(f"gold_step_{i}_not_object")
            continue
        tool_name = step.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            errors.append(f"gold_step_{i}_missing_tool")
            continue
        if tool_name not in tmap:
            errors.append(f"gold_step_{i}_unknown_tool:{tool_name}")
            continue
        if tool_name not in domain_tool_names:
            errors.append(f"gold_step_{i}_cross_domain:{tool_name}:not_in_{domain}")
        # ── search-style steps must have matching ref type
        required_types = _SEARCH_TOOL_REF_TYPES.get(tool_name)
        if required_types is not None and not (required_types & ref_types_present):
            errors.append(
                f"gold_step_{i}_no_ref_coverage:{tool_name} "
                f"(needs ref of type ∈ {sorted(required_types)}; refs have {sorted(ref_types_present)})"
            )
        # ── argument-level checks
        args = step.get("arguments") or {}
        if not isinstance(args, dict):
            errors.append(f"gold_step_{i}_args_not_object")
            continue
        for ak, av in args.items():
            # autofill should be omitted (env injects)
            if ak in _RUNTIME_AUTOFILLED:
                errors.append(
                    f"gold_step_{i}_arg_uses_autofill:{ak} "
                    "(omit from arguments; env auto-injects)"
                )
            # id-shaped args must reference real refs (top-level ref.id
            # OR id-shaped string in any ref's attributes — e.g.
            # message.attributes.thread_id is a legitimate target for
            # draft_message.thread_id).
            if ak.endswith("_id") and isinstance(av, str):
                if not _looks_like_id_value(av):
                    # placeholder like '<msg id>' — skip strict check
                    continue
                if av not in valid_id_pool:
                    errors.append(
                        f"gold_step_{i}_unknown_ref_id:{ak}={av!r} "
                        f"(not in local_env.references nor any ref's attributes)"
                    )
            elif ak.endswith("_ids") and isinstance(av, list):
                for v in av:
                    if isinstance(v, str) and _looks_like_id_value(v) and v not in valid_id_pool:
                        errors.append(
                            f"gold_step_{i}_unknown_ref_id_in_list:{ak} item {v!r}"
                        )
    return errors


def _is_existing_session_valid(out_path: Path) -> bool:
    """Whether an already-written session file still matches the current
    schema. Used by run_persona (default non-force mode) to decide which
    skeletons need regeneration.

    Pydantic reconstruction is the canonical check — it catches schema
    drift. Schema: task_params is a flat dict[str, Any]; a
    {value, desc} shape is rejected as malformed (the dict-typed value
    fails any downstream consumer that expects a scalar).
    """
    if not out_path.exists():
        return False
    try:
        LearningSession(**read_json(out_path))
    except Exception:
        return False
    return True


def _validate_fill(fill: dict, domain: str, tmap: dict[str, dict]) -> list[str]:
    errors: list[str] = []

    for field in ("reason_for_call", "task_params", "local_env", "gold_trajectory"):
        v = fill.get(field)
        if v is None or (isinstance(v, str) and not v.strip()) or (
            isinstance(v, (list, dict)) and not v
        ):
            errors.append(f"missing:{field}")
    if errors:
        return errors

    # ── task_params (flat dict: {key: concrete_value}) ────────────────────
    tp = fill["task_params"]
    if not isinstance(tp, dict) or not tp:
        errors.append("task_params_empty")
        return errors

    # Quality floor #3: field count band
    if len(tp) < _TASK_PARAMS_MIN:
        errors.append(f"task_params_too_few:{len(tp)}<{_TASK_PARAMS_MIN}")
    if len(tp) > _TASK_PARAMS_MAX:
        errors.append(f"task_params_too_many:{len(tp)}>{_TASK_PARAMS_MAX}")

    # Reject the {value, desc} wrapper shape. task_params values
    # should be the raw concrete value (str / int / list / ...), not a dict.
    # A dict with a "value" key is malformed and needs regeneration.
    for k, v in tp.items():
        if isinstance(v, dict) and "value" in v:
            errors.append(f"task_param_legacy_wrapper:{k} (flatten to raw value)")

    # Collect ref ids once (used by id-coverage check below).
    local_env = fill["local_env"]
    if not isinstance(local_env, dict):
        errors.append("local_env_not_object")
        return errors

    refs = local_env.get("references") or []
    if not isinstance(refs, list) or not refs:
        errors.append("local_env_references_empty")
        refs = []

    allowed_types = ref_types_for_domain(domain)
    ref_ids: set[str] = set()
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            errors.append(f"ref_{i}_not_object")
            continue
        rid = r.get("id")
        rtype = r.get("type")
        attrs = r.get("attributes")
        if not isinstance(rid, str) or not rid.strip():
            errors.append(f"ref_{i}_missing_id")
            continue
        if rid in ref_ids:
            errors.append(f"ref_{i}_duplicate_id:{rid}")
        ref_ids.add(rid)
        if rtype not in allowed_types:
            errors.append(f"ref_{i}_invalid_type:{rtype}:domain={domain}")
        if not isinstance(attrs, dict):
            errors.append(f"ref_{i}_attributes_not_object")
        # Also include id-shaped values from ref attributes (e.g.
        # message.attributes.thread_id, product.attributes.order_id)
        # so derived-entity IDs pass the task_params coverage check.
        elif isinstance(attrs, dict):
            for av in attrs.values():
                if isinstance(av, str) and _looks_like_id_value(av):
                    ref_ids.add(av)
                elif isinstance(av, list):
                    for item in av:
                        if isinstance(item, str) and _looks_like_id_value(item):
                            ref_ids.add(item)

    # Quality floor #1 (HARD): task_param id-ish values must exist in refs,
    # EXCEPT: (a) runtime-autofilled identity fields (contact_info etc.), or
    # (b) env auto-preseed reference ids (currently none — see _AUTO_PRESEED_IDS).
    for k, val in tp.items():
        if k in _RUNTIME_AUTOFILLED:
            continue
        for id_val in _extract_id_values(k, val):
            if id_val in ref_ids or id_val in _AUTO_PRESEED_IDS:
                continue
            if k in _RESULT_OBJECT_ID_KEYS:
                # Result-object keys have no allowed ref type — the only
                # legitimate fix is to rewrite as a description field.
                noun = k[: -len("_id")] if k.endswith("_id") else k
                errors.append(
                    f"task_param_id_not_in_refs:{k}={id_val} "
                    f"({k} refers to a result of a past tool call — current "
                    f"ref schema has no {noun} ref type. Replace this with "
                    f"`{noun}_description: \"...\"` (a free-text descriptor "
                    f"the agent can use to track_*/search_* at runtime). "
                    f"Do NOT add a {noun} object to local_env.references.)"
                )
            else:
                errors.append(
                    f"task_param_id_not_in_refs:{k}={id_val} "
                    "(add a matching object to local_env.references, "
                    "or replace with a descriptive field the agent can search)"
                )

    # Quality floor #2 (HARD): output-like fields cannot carry long prewritten text
    for k, val in tp.items():
        if _is_output_like_field(k) and isinstance(val, str) and len(val) > _OUTPUT_VALUE_MAX_CHARS:
            errors.append(
                f"task_param_preauthored_output:{k} "
                f"(value length {len(val)} > {_OUTPUT_VALUE_MAX_CHARS}; "
                "keep user-side as points/keywords and let the agent draft)"
            )

    # ── Cheap autofill check (E from the design ideal) ──────────────────
    errors.extend(_validate_autofill(tp))

    # ── Domain-specific intra-session consistency ─────────────────────────
    # scheduling: task_params time fields must match the referenced event's
    # actual times in local_env.  Mismatch means the user_sim and the agent
    # will work from contradictory clocks (different start/end times for the
    # same calendar event).
    if domain == "scheduling":
        title = tp.get("title")
        if title:
            ev_match = next(
                (r for r in refs
                 if isinstance(r, dict) and r.get("attributes", {}).get("title") == title),
                None,
            )
            if ev_match is None:
                # Atomic constraint warning: the title/start_time/end_time
                # triplet must match the SAME calendar_event ref. Refining
                # only the title without syncing time fields will trip
                # scheduling_time_mismatch on the next pass.
                errors.append(
                    f"scheduling_event_not_found: task_params.title='{title}' "
                    "matches no calendar_event in local_env.references. "
                    "Atomic fix: pick a calendar_event ref AND set "
                    "task_params.title / start_time / end_time to that ref's "
                    "attributes.title / start_time / end_time simultaneously "
                    "(or change the ref's attributes to match task_params); "
                    "do not change one without syncing the others."
                )
            else:
                ev_attrs = ev_match.get("attributes", {})
                for time_key in ("start_time", "end_time"):
                    tp_val = tp.get(time_key)
                    ev_val = ev_attrs.get(time_key)
                    if tp_val and ev_val and tp_val != ev_val:
                        errors.append(
                            f"scheduling_time_mismatch:{time_key} "
                            f"task_params='{tp_val}' != "
                            f"local_env[{ev_match['id']}]='{ev_val}' — "
                            "either set task_params.{time_key} to the ref's "
                            "value, or update the ref's attributes.{time_key} "
                            "to the task_params value, but the two must agree."
                        )

    # ── gold_trajectory check (subsumes the previous F / B / id-coverage
    # closed-loop checks: a coherent oracle trajectory IS the proof of
    # internal consistency between task_params, refs, and tools).
    errors.extend(
        _validate_gold_trajectory(
            fill.get("gold_trajectory") or [],
            tp, refs, domain, tmap,
        )
    )
    errors.extend(
        _validate_write_args_grounded_in_task_params(
            fill.get("gold_trajectory") or [],
            tp, tmap,
        )
    )
    errors.extend(
        _validate_search_args_grounded_in_task_params(
            fill.get("gold_trajectory") or [],
            tp,
        )
    )

    # ── Gold trajectory dry-run — only when static checks passed ─────────
    # Executes each gold step against a live env instance to catch runtime
    # filter failures that static token-overlap checks can't see (e.g.
    # a search tool returning 0 results because the task_param value doesn't
    # match any ref attribute at the env's filter granularity).
    if not errors:
        errors.extend(_dry_run_gold_trajectory(fill, domain))

    return errors


def _collect_ids_from_result(obj: Any) -> set[str]:
    """Recursively extract all id-like strings from a tool result object."""
    out: set[str] = set()
    if isinstance(obj, str) and _looks_like_id_value(obj):
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _collect_ids_from_result(v)
    elif isinstance(obj, list):
        for item in obj:
            out |= _collect_ids_from_result(item)
    return out


def _dry_run_gold_trajectory(fill: dict, domain: str) -> list[str]:
    """Execute gold_trajectory against a live env and verify runtime reachability.

    Complements static checks by catching what only execution can reveal:

    1. Tool execution errors (bad arg types, env rejects call)
    2. Search steps returning 0 results (filter mismatch at runtime)
    3. Action step *_id args not reachable from prior search results —
       the ID exists in refs (static already confirmed), but the search
       filters don't surface it, so a real agent following this trajectory
       would never discover the ID.

    Seed pool for check 3: task_params *_id values are included so that
    pre-selected IDs (product_id, selected_stop_ids, etc.) pass without
    requiring a prior search step.

    Uses a dummy PersonaProfile — identity autofill fields don't affect
    trajectory correctness. Lazy-imports runner.environment to avoid
    circular-import risk at module load time.
    """
    from types import SimpleNamespace

    from runner.environment.base import PersonaProfile
    from runner.environment.domains import build_env_for_session
    from runner.schemas import LearningSession as _LS

    gold = fill.get("gold_trajectory") or []
    if not gold:
        return []

    raw_env = fill.get("local_env") or {}
    try:
        session = _LS(
            session_id="__validation__",
            session_type="learning",
            day_offset=0,
            domain=domain,
            reason_for_call=fill.get("reason_for_call") or "",
            task_params=fill.get("task_params") or {},
            expected_tools=list(
                dict.fromkeys(
                    s["tool"] for s in gold if isinstance(s, dict) and "tool" in s
                )
            ),
            gold_trajectory=gold,
            local_env={
                "tools": [t["name"] for t in domain_tools(domain)],
                "references": raw_env.get("references") or [],
            },
        )
    except Exception as e:
        return [f"dry_run_session_build_failed:{e}"]

    persona = PersonaProfile(
        persona_id="__validation__",
        default_shipping_address="10 Main St, New York, NY 10001",
        default_payment_method="CARD_VAL001",
        default_contact="validation@example.com",
    )

    try:
        env = build_env_for_session(session_task=session, persona=persona)
    except Exception as e:
        return [f"dry_run_env_build_failed:{e}"]

    reachable_ids: set[str] = set()
    # Seed from task_params so user-provided ids (product_id,
    # selected_stop_ids, etc.) pass without requiring a prior search.
    for k, v in (fill.get("task_params") or {}).items():
        for id_val in _extract_id_values(k, v):
            reachable_ids.add(id_val)

    errors: list[str] = []
    for i, step in enumerate(gold):
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        arguments = step.get("arguments") or {}
        call = SimpleNamespace(
            name=tool_name, arguments=arguments,
            id=f"val_{i}", requestor="assistant",
        )
        resp = env.get_response(call)

        if resp.is_error:
            errors.append(f"dry_run_step_{i}_{tool_name}_error: {resp.content}")
            continue

        try:
            result = json.loads(resp.content)
        except Exception:
            continue

        is_search = tool_name in _SEARCH_TOOL_REF_TYPES
        if is_search:
            if isinstance(result, dict) and result.get("count") == 0:
                errors.append(
                    f"dry_run_step_{i}_{tool_name}_empty_results "
                    f"(args: {json.dumps(arguments, ensure_ascii=False)})"
                )
                continue

        # Always collect IDs from results so chained write→modify
        # sequences work (e.g. book_restaurant → modify_restaurant_reservation).
        reachable_ids |= _collect_ids_from_result(result)

        if not is_search:
            # Action step: *_id args must have been surfaced by a prior
            # search or created by a prior write step.
            for k, v in arguments.items():
                if k.endswith("_id") and isinstance(v, str) and _looks_like_id_value(v):
                    if v not in reachable_ids:
                        errors.append(
                            f"dry_run_step_{i}_{tool_name}: "
                            f"{k}='{v}' not returned by any prior search step"
                        )
                elif k.endswith("_ids") and isinstance(v, list):
                    for item in v:
                        if (
                            isinstance(item, str)
                            and _looks_like_id_value(item)
                            and item not in reachable_ids
                        ):
                            errors.append(
                                f"dry_run_step_{i}_{tool_name}: "
                                f"{k}[] '{item}' not returned by any prior search step"
                            )

    return errors


# ---------------------------------------------------------------------------
# LLM call + assembly
# ---------------------------------------------------------------------------


def _assemble_session(skeleton: dict, domain: str, result: dict) -> dict:
    """Assemble a validated LLM fill result into the final LearningSession dict."""
    full_domain_tools = [t["name"] for t in domain_tools(domain)]
    gold_traj = result["gold_trajectory"]
    seen: set[str] = set()
    derived_expected_tools: list[str] = []
    for step in gold_traj:
        t = step.get("tool")
        if isinstance(t, str) and t not in seen:
            seen.add(t)
            derived_expected_tools.append(t)
    return {
        "session_id": skeleton["session_id"],
        "session_type": "learning",
        "day_offset": skeleton["day_offset"],
        "domain": domain,
        "reason_for_call": result["reason_for_call"].strip(),
        "task_params": result["task_params"],
        "expected_tools": derived_expected_tools,
        "gold_trajectory": [
            {"tool": s["tool"], "arguments": s.get("arguments") or {}}
            for s in gold_traj
        ],
        "local_env": {
            "tools": full_domain_tools,
            "references": result["local_env"]["references"],
        },
    }


_REFINE_INSTRUCTIONS = """\
---

## Previous attempt

Your earlier JSON output (below) failed validation. Refine it — do **not** start over.

```json
{prev_json}
```

## Validation errors

{error_list}

## How to refine

- **Minimal edits only.** Touch *only* the fields implicated by the errors above; keep every other field exactly as you wrote it. Do not rewrite well-formed `task_params` / `references` / `gold_trajectory` entries.
- Each error is a hard rejection — the field it points to must change in a way that clears the message.
- After editing, re-check internal consistency: every `*_id` value in `task_params` and every id argument in `gold_trajectory` must still resolve to a real ref in `local_env.references`; every search-style step in `gold_trajectory` must still have ≥1 ref of the matching type.
- Output the **complete corrected JSON** with the same top-level schema (`reason_for_call` / `task_params` / `local_env` / `gold_trajectory`). No markdown fence, no extra text.
"""


def _refine_fill(
    base_prompt: str,
    prev_result: dict,
    prev_errors: list[str],
    domain: str,
    tmap: dict[str, dict],
    model: str,
) -> tuple[dict | None, list[str]]:
    """One minimal-edit refine pass.

    Re-uses the original fill prompt body verbatim (so every constraint
    that drove validation remains in scope), then appends the previous
    JSON output + the validation errors + a "minimal edits" directive so
    the LLM produces a diff-style fix rather than a fresh rewrite.
    Returns `(refined_result, [])` on success, or `(None, errors)`
    where errors are namespaced with `refine_*:` / `refine:` prefixes
    distinguishing infra failures from validation failures.
    """
    appendix = _REFINE_INSTRUCTIONS.format(
        prev_json=json.dumps(prev_result, ensure_ascii=False, indent=2),
        error_list="\n".join(f"- {e}" for e in prev_errors),
    )
    refine_prompt = base_prompt + "\n\n" + appendix

    try:
        result = call_llm_json(refine_prompt, model=model, temperature=0.3, max_retries=3)
    except Exception as e:
        return None, [f"refine_llm_error:{e}"]
    if not isinstance(result, dict):
        return None, [f"refine_non_dict:{type(result).__name__}"]

    errors = _validate_fill(result, domain, tmap)
    if errors:
        return None, [f"refine:{e}" for e in errors]
    return result, []


def fill_one(
    skeleton: dict,
    structured: dict,
    model: str,
) -> tuple[dict | None, list[str], dict | None]:
    """Generate one LearningSession; refine once on validation failure.

    Learning-session skeleton hands over only `{domain, theme_one_line}`.
    Learning-session fill owns the rest:
    `reason_for_call` / `task_params` / `references` / `gold_trajectory`
    are all LLM-generated together so the tool set commits to whatever
    the content actually exercises (no upfront skeleton commitment to fight).

    Returns `(session, errors, debug)`:
      - success:                    `(session, [], None)`
      - LLM crash before any output: `(None, [<error>], None)`
      - validation+refine fail:     `(None, [<refine errors>], debug)` where
        `debug = {"first_result": ..., "first_errors": [...]}` so the
        offline `fill_failures.json` records what the LLM actually
        wrote, not just the error tags.
    """
    domain = skeleton["domain"]
    template = load_prompt(PROMPTS_DIR, name="gen")
    base_prompt = fill_prompt(
        template,
        SKELETON=skeleton,
        STRUCTURED_PERSONA=format_structured_persona(structured),
        DOMAIN=domain,
        DOMAIN_TOOLS=render_domain_tools(domain),
        REF_SCHEMA=render_ref_schema(domain),
    )
    tmap = tool_map()

    try:
        first_result = call_llm_json(base_prompt, model=model, temperature=0.6, max_retries=3)
    except Exception as e:
        return None, [f"llm_error:{e}"], None

    if not isinstance(first_result, dict):
        return None, [f"non_dict_response:{type(first_result).__name__}"], None

    first_errors = _validate_fill(first_result, domain, tmap)
    if not first_errors:
        return _assemble_session(skeleton, domain, first_result), [], None

    logger.info(
        "[%s] first fill failed (%d errors); attempting refine",
        skeleton.get("session_id"), len(first_errors),
    )
    refined, refine_errors = _refine_fill(
        base_prompt, first_result, first_errors, domain, tmap, model,
    )
    if refined is not None:
        logger.info("[%s] refine succeeded", skeleton.get("session_id"))
        return _assemble_session(skeleton, domain, refined), [], None

    debug = {"first_result": first_result, "first_errors": first_errors}
    return None, refine_errors, debug


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_persona(
    persona_id: str,
    session_id_filter: str | None,
    model: str,
    force: bool,
) -> dict:
    pdir = PERSONAS_DIR / persona_id
    skeleton_path = pdir / "skeleton.json"
    structured_path = pdir / "structured.json"
    if not skeleton_path.exists() or not structured_path.exists():
        logger.error("[%s] missing skeleton.json or structured.json", persona_id)
        return {"persona_id": persona_id, "status": "missing_inputs"}

    skeletons = read_json(skeleton_path)
    structured = read_json(structured_path)

    if session_id_filter:
        skeletons = [s for s in skeletons if s["session_id"].startswith(session_id_filter)]

    out_dir = pdir / "learning_sessions"

    # --force protocol: archive previous outputs to *_prev/ slots before
    # regen. Lets us compare rounds and recover a known-good prior state.
    # Empty sources (e.g. dir already cleared by an upstream skeleton
    # --force) are removed without clobbering an existing prev slot.
    if force:
        archived: list[str] = []
        for p in (out_dir, pdir / "fill_failures.json"):
            bak = archive_to_prev(p)
            if bak is not None:
                archived.append(f"{p.name}→{bak.name}")
        if archived:
            logger.info("[%s] archived to *_prev: %s", persona_id, ", ".join(archived))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Schema-aware retry (only relevant when not --force):
    #   skip sessions that are already written AND still match the current
    #   LearningSession schema; stale/missing ones are deleted + regenerated.
    #   Saves LLM cost on prompt tweaks that don't invalidate existing outputs.
    if force:
        # learning_sessions/ has been archived above and recreated empty;
        # all skeletons need regen.
        pass
    else:
        before = len(skeletons)
        needs_gen: list[dict] = []
        for s in skeletons:
            p = out_dir / f"{s['session_id']}.json"
            if _is_existing_session_valid(p):
                continue
            if p.exists():
                p.unlink()  # delete stale so we don't leave old data behind
            needs_gen.append(s)
        skeletons = needs_gen
        skipped = before - len(skeletons)
        if skipped:
            logger.info(
                "[%s] skipped %d schema-valid sessions; regenerating %d invalid/missing",
                persona_id, skipped, len(skeletons),
            )

    if not skeletons:
        logger.info("[%s] nothing to fill", persona_id)
        return {"persona_id": persona_id, "status": "nothing_to_do"}

    logger.info(
        "[%s] filling %d sessions, model=%s (concurrency capped by llm.PER_MODEL_MAX_CONCURRENCY)",
        persona_id, len(skeletons), model,
    )

    stats = {"ok": 0, "fail": 0}
    failures: list[dict] = []
    t0 = time.time()

    # Persona-level bucket shared across fill workers; each worker re-enters
    # model_scope with this same reference so its call_llm_json attributes
    # to the persona's learning_session slot.
    token_bucket = make_bucket()

    def _fill_with_scope(sk: dict) -> tuple[dict | None, list[str], dict | None]:
        with model_scope(token_bucket):
            return fill_one(sk, structured, model)

    with ThreadPoolExecutor(max_workers=max(1, len(skeletons))) as pool:
        futs = {pool.submit(_fill_with_scope, sk): sk for sk in skeletons}
        for fut in as_completed(futs):
            sk = futs[fut]
            sid = sk["session_id"]
            try:
                session, errors, debug = fut.result()
            except Exception as e:
                stats["fail"] += 1
                failures.append({"session_id": sid, "error": str(e)})
                logger.error("[%s] %s crashed: %s", persona_id, sid, e)
                continue
            if session is None:
                stats["fail"] += 1
                failure = {"session_id": sid, "errors": errors, "skeleton": sk}
                if debug is not None:
                    # Refine ran but didn't save the session. Preserve what
                    # the LLM actually wrote so offline debug doesn't have
                    # to guess from error tags alone.
                    failure["first_result"] = debug.get("first_result")
                    failure["first_errors"] = debug.get("first_errors")
                failures.append(failure)
                logger.warning("[%s] %s rejected: %s", persona_id, sid, errors)
                continue
            write_json(out_dir / f"{sid}.json", session)
            stats["ok"] += 1
            logger.info("[%s] %s filled (tools=%d, refs=%d)",
                        persona_id, sid,
                        len(session["local_env"]["tools"]),
                        len(session["local_env"]["references"]))

    if failures:
        write_json(pdir / "fill_failures.json", failures)

    record_stage(pdir, "learning_session", token_bucket)

    elapsed = time.time() - t0
    logger.info("[%s] done in %.1fs. ok=%d fail=%d",
                persona_id, elapsed, stats["ok"], stats["fail"])
    return {
        "persona_id": persona_id,
        "status": "ok",
        "filled": stats["ok"],
        "failed": stats["fail"],
    }


def main():
    parser = argparse.ArgumentParser(description="Learning-session fill: fill learning sessions")
    parser.add_argument("--persona-id", type=str, default=None,
                        help="prefix match; omit to fill all personas")
    parser.add_argument("--session-id", type=str, default=None,
                        help="only fill sessions whose id startswith this")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_pids = sorted(
        p.name for p in PERSONAS_DIR.iterdir()
        if p.is_dir() and (p / "skeleton.json").exists()
    )
    if args.persona_id:
        all_pids = [p for p in all_pids if p.startswith(args.persona_id)]
        if not all_pids:
            parser.error(f"no persona matched --persona-id {args.persona_id!r}")

    if not all_pids:
        logger.info("nothing to do.")
        return

    logger.info("processing %d personas", len(all_pids))
    for pid in all_pids:
        run_persona(pid, args.session_id, args.model, args.force)


if __name__ == "__main__":
    main()
