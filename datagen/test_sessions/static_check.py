"""Static QC for generated TestSession (before runtime). Adapted to the
flat 4-branch check_type schema (tool_identity / param_id / param_enum /
confirm); every gold is one tool call. confirm rules emit a single
get_user_confirmation call (no chain trailing the mutate).

Zero-LLM, deterministic. Run after every LLM generation; reject + ask for
regeneration on failure.
"""
from __future__ import annotations

import re
from typing import Any

from datagen._common import (
    DOMAINS,
    base_tool_names,
    domain_tools,
    ref_types_for_domain,
    rule_is_permission,
    tool_map,
)

# ---------------------------------------------------------------------------
# Reference-type-per-domain — dynamically resolved from TableSpec.
# See datagen._common.ref_types_for_domain().
# ---------------------------------------------------------------------------

# Tools that search catalogs (no need to pre-declare refs for results).
_SEARCH_TYPE_TOOLS: set[str] = {
    "search_products", "search_events", "search_trip_stops",
    "search_hotels", "search_flights", "search_ground_transport",
    "search_restaurants", "search_service_providers", "search_destinations",
    "list_events", "list_files", "search_documents", "search_messages",
}

# Exploration / read-only tools — these MUST NOT be the decision step in gold,
# because passive agents commonly call them as part of normal commonsense
# exploration ("first I'll search to compare options...") and the
# ordered-subsequence trace match would then trivially count the passive trace
# as hitting gold even though the actual decision (the booking) was something
# else. Gold should always be a final-action tool (book_/cancel_/place_/
# archive_/delete_/set_/update_/respond_/classify_/label_/move_/...).
#
# Allowed as the FIRST step of a chain (a chain can legitimately prologue with
# search/list and then act); enforced only on the LAST step of a chain (the
# decision step) and on the only step of a single-action rule.
#
# `find_` is deliberately absent: the only find_* tool in the ontology is
# `find_alternative_slots`, which is a legitimate decision-step tool for
# fallback-style rules. track_*/notify_*/set_reminder/label_messages/
# update_tracker/review_recurring_charges don't match any prefix here and
# are naturally valid decision-step tools.
#
# NOTE — this set defines what CANNOT be a gold decision step (last step
# of chain / only step of single). It is intentionally NOT identical to
# `evaluator/action_match._READ_PREFIXES` (which controls chain-prologue
# substitution at runtime); that set is broader (`check_/track_/review_`)
# because those tools can legitimately end a chain. Keep this set focused
# on the binding-side guard. `binding._EXPLORATION_TOOL_PREFIXES` mirrors
# this list exactly — they enforce the same invariant on different
# pipelines.
_EXPLORATION_TOOL_PREFIXES: tuple[str, ...] = (
    "search_", "list_", "lookup_", "get_", "view_", "browse_", "compare_",
)

# Auto-preseeded ids the runtime adds to every session regardless of refs.
# (Communication's priority slot is a flat enum value space, so no
# prefix is auto-injected.)
_AUTO_PRESEED_IDS: set[str] = set()

# Enum values that signal a rule direction — never allowed to appear verbatim
# in the instruction text (would leak the rule to passive agents).
_LEAKY_ENUM_VALUES = {
    # reservation/travel attribute-style values (from old ref pool)
    "local_adapted", "established", "novelty", "popular", "value",
    # book_flight.seat_preference + book_restaurant table preferences
    "window", "aisle", "middle", "quiet", "outdoor_patio", "bar_counter",
    # classify_files.scheme
    "by_project", "by_type", "by_date", "by_owner", "by_priority",
    # return_order.resolution_type
    "refund", "exchange", "store_credit",
    # respond_to_event_invite.response
    "accept", "decline", "tentative",
    # book_event_ticket.ticket_type
    "general_admission", "senior", "vip", "child",
    # replan_trip.replacement_type
    "swap_single_stop", "swap_all_stops",
    # create_document.doc_type
    "memo", "spreadsheet", "report", "template",
    # set_message_priority.priority
    "high", "medium", "low", "normal",
    # set_reminder.cadence
    "once", "daily", "weekly", "monthly", "before_event",
    # update_tracker.update_action
    "create_entry", "update_status", "record_next_action", "close_entry",
    # change_subscription_plan.new_plan
    "basic", "pro", "premium", "enterprise",
}

# Natural-language synonyms for enum gold values — leak vectors that must
# never appear in instruction text. Each entry maps the enum value to
# substrings that strongly imply that value; if the gold value is present
# in required_actions AND any of its synonyms appears in instruction
# (case-insensitive substring match), flag it as a leak.
#
# Substrings, not regex — kept high-signal so the leak detector doesn't
# misfire on incidental wording. Phrases like "by project" are unambiguous
# leaks for `by_project`; the bare word "project" alone is NOT in the
# table because it appears naturally in many neutral contexts (e.g.
# "documents from this project").
_LEAKY_ENUM_SYNONYMS: dict[str, tuple[str, ...]] = {
    # workspace.classify_files.scheme
    "by_project":     ("by project", "by-project", "per project", "per-project",
                       "per client", "per-client", "by client", "by-client",
                       "by workstream", "by work stream", "by matter"),
    "by_date":        ("by date", "by-date", "chronologically", "by month",
                       "by year", "in date order", "date-based",
                       "in chronological"),
    "by_type":        ("by type", "by-type", "by file type", "by format",
                       "by category"),
    "by_owner":       ("by owner", "by-owner", "by author", "by who"),
    "by_priority":    ("by priority", "by-priority", "by importance",
                       "by urgency"),
    # scheduling.respond_to_event_invite.response
    "accept":         ("just accept", "go ahead and accept", "say yes",
                       "rsvp yes", "accept it", "accept the invite",
                       "i'll go", "i can make it", "count me in"),
    "decline":        ("just decline", "say no", "rsvp no", "decline it",
                       "i'll pass", "turn it down", "skip it",
                       "decline the invite"),
    "tentative":      ("tentative", "maybe", "hold the slot",
                       "keep my options", "keep flexible",
                       "leave it open", "tentatively",
                       "rsvp tentative"),
    # commerce.return_order.resolution_type
    "refund":         ("refund", "money back", "return for cash"),
    "exchange":       ("exchange", "swap it", "trade for another"),
    "store_credit":   ("store credit", "credit", "voucher"),
    # reservation/travel attribute synonyms (from old pool)
    "local_adapted":  ("locally adapted", "local style", "regional",
                       "native-style", "authentic local"),
    "established":    ("established", "classic", "well-known",
                       "long-running", "traditional"),
    "novelty":        ("novelty", "trendy", "newest", "up-and-coming",
                       "buzzy", "experimental"),
    "popular":        ("most popular", "trending", "top-rated",
                       "highest rated", "best reviewed"),
    "value":          ("best value", "good value", "cheap option",
                       "most affordable", "budget pick"),
    # travel.book_flight.seat_preference
    "window":         ("window seat", "by the window", "next to the window"),
    "aisle":          ("aisle seat", "on the aisle"),
    "middle":         ("middle seat",),
    # other freeform-but-leaky enum values
    "quiet":          ("quiet spot", "quiet table", "quiet corner",
                       "secluded", "tucked away"),
    # scheduling.set_reminder.cadence
    "weekly":         ("weekly", "every week", "once a week",
                       "every monday", "every tuesday", "every wednesday",
                       "every thursday", "every friday", "every saturday",
                       "every sunday"),
    "daily":          ("daily", "every day", "once a day", "each day"),
    "monthly":        ("monthly", "every month", "once a month",
                       "each month"),
    "once":           ("just once", "one time", "only once",
                       "single reminder", "one-time"),
    "before_event":   ("before the event", "right before", "ahead of the event",
                       "prior to the event", "leading up to the event"),
    # commerce.change_subscription_plan.new_plan
    "basic":          ("basic plan", "basic tier", "downgrade to basic",
                       "starter plan", "entry level"),
    "pro":            ("pro plan", "pro tier", "upgrade to pro",
                       "professional plan"),
    "premium":        ("premium plan", "premium tier", "upgrade to premium",
                       "go premium"),
    "enterprise":     ("enterprise plan", "enterprise tier",
                       "upgrade to enterprise"),
    # workspace.create_document.doc_type — bare "note" is too high-noise to
    # add to _LEAKY_ENUM_VALUES; only the action phrasings get blocked.
    "note":           ("create a note", "write a note", "jot down",
                       "quick note", "make a note"),
}

# Attribute names that evaluatively hint the rule direction — reject on sight.
_EVALUATIVE_ATTRS = {
    "recommendation_signal", "suitability", "user_favorite", "highlights",
    "importance_hint", "broader_signal", "priority_hint", "recommended_action",
    "flexibility", "moveable", "can_reschedule", "is_flexible", "is_optional",
    "removable", "suggested_choice",
}

# Tools that operate on secondary entities (orders, trips, appointments,
# reservations, bookings) — their *_id arguments reference runtime objects
# the env creates during hydration from reference attributes.  When a gold
# action targets such an entity, at least one reference of the corresponding
# type must carry the entity ID in its attributes so the env can hydrate it.
# Built dynamically from TableSpec declarations (single source of truth).


def _build_secondary_entity_id_map() -> dict[str, dict[str, str]]:
    from runner.environment.tables import all_table_specs as _ats
    out: dict[str, dict[str, str]] = {}
    for _domain, spec in _ats():
        if spec.kind != "derived":
            continue
        if not spec.source_ref_type or not spec.source_attr:
            continue
        for tool in spec.operating_tools:
            out.setdefault(tool, {})[spec.source_attr] = spec.source_ref_type
    return out


_SECONDARY_ENTITY_ID_MAP: dict[str, dict[str, str]] = _build_secondary_entity_id_map()

# Param names handled by check 11 — check 5 should skip these to avoid duplicate errors.
_SECONDARY_ENTITY_PARAMS: set[str] = set()
for _m in _SECONDARY_ENTITY_ID_MAP.values():
    _SECONDARY_ENTITY_PARAMS.update(_m.keys())

_PROACTIVE_TRIGGER_RE = re.compile(
    r"keep (an )?eye on|keep tabs on|track closely|"
    r"\bmonitor\b|monitoring|notify me when|notify me if|"
    r"let me know when|let me know if (anything|something) (changes|goes|happens)|"
    r"watch out for|watch for|keep watching|stay on top of|"
    r"proactively|proactive|"
    r"ping me when|ping me if|alert me when|alert me if",
    re.IGNORECASE,
)


# Reference-attribute keys that runtime search/list tools use as filter
# input via _loose_string_match. If a gold-referenced ref carries one of
# these attributes, the instruction must contain at least one token that
# loose-matches the attribute value — otherwise no search query the agent
# can derive from the instruction will surface the gold ref, and the
# session is unsolvable even by oracle.
#
# Mirrors the runtime filter-input parameters across domains:
#   reservation: search_restaurants(location/cuisine), search_events(location),
#                search_service_providers(service_type/location)
#   travel:      search_destinations(destination_constraints → tags/name/region),
#                search_trip_stops(destination → name/location, stop_tags),
#                search_flights/hotels/ground_transport(origin/destination/location)
#   communication: search_messages(sender/folder)
#   workspace:   list_files(location/file_type), search_documents(location/doc_type)
#   commerce:    search_products(category/budget) — category-only string match
_SEARCHABLE_ATTR_KEYS: tuple[str, ...] = (
    "name", "title", "subject", "sender",
    "location", "region", "city", "address",
    "destination", "origin", "route",
    "cuisine", "service_type", "category", "doc_type",
    "tags", "event_tags", "stop_tags",
)


# Loose token-level intersection identical in spirit to runtime
# `_loose_string_match`. We re-implement here (rather than importing from
# `runner.environment.base`) because (a) datagen shouldn't depend on
# runner internals, and (b) the runtime function returns True on empty
# query, which we don't want during static analysis.
def _instruction_loose_hits(instruction: str, target: str) -> bool:
    if not instruction or not target:
        return False
    instr_l = instruction.lower()
    tgt_l = target.lower().strip()
    if not tgt_l:
        return False
    if tgt_l in instr_l:
        return True
    instr_tokens = set(re.split(r"[\s,./\-_:'\"()\[\]【】，。！？；：、]+", instr_l)) - {""}
    tgt_tokens = set(re.split(r"[\s,./\-_:'\"()\[\]【】，。！？；：、]+", tgt_l)) - {""}
    # Drop stop tokens (length < 2 mostly catches articles / digits-as-noise)
    instr_tokens = {t for t in instr_tokens if len(t) >= 2}
    tgt_tokens = {t for t in tgt_tokens if len(t) >= 2}
    return bool(instr_tokens & tgt_tokens)


def _ref_searchable_values(ref: dict) -> list[tuple[str, str]]:
    """Return [(attr_key, value), ...] for searchable attributes whose
    values are non-empty strings (or list[str] flattened item-wise)."""
    attrs = ref.get("attributes") or {}
    out: list[tuple[str, str]] = []
    # Top-level "name" sometimes lives on the ref itself, but our schema
    # keeps it under attributes — only check attributes here.
    for k in _SEARCHABLE_ATTR_KEYS:
        v = attrs.get(k)
        if isinstance(v, str) and v.strip():
            out.append((k, v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.append((k, item))
    return out


# ── Time consistency helpers (MVP-no-time-axis policy) ─────────────────
#
# Search tools no longer accept time filters; agent reads each ref's
# date_time field after fetching. Therefore:
#  - Default: instruction does NOT contain time. Skip time check.
#  - If instruction contains an absolute date (YYYY-MM-DD), gold ref's
#    date fields must include that date as a substring.
#  - Relative-time phrases are forbidden (agent can't resolve them; gold
#    won't align). Soft-detect a few common ones for diagnostics.

_ABS_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Relative time phrases — these are leak vectors: an instruction that
# says "next Thursday" implies a date the agent can't compute, so the
# ref's actual date drifts off. Keep this list short and high-signal.
_RELATIVE_TIME_PATTERNS = (
    re.compile(r"\bnext\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b|\btonight\b|\bin a few days\b", re.IGNORECASE),
)

# Reference attribute keys that hold ref's effective date/time signal.
# Time-consistency check looks at these for substring match against
# instruction's absolute date tokens.
_REF_DATE_ATTR_KEYS: tuple[str, ...] = (
    "date_time", "start_time", "end_time",
    "departure_date", "return_date",
    "check_in_date", "check_out_date",
    "timestamp", "scheduled_at",
)


def _ref_date_values(ref: dict) -> list[str]:
    """Return [string]: every date-bearing attribute value from a ref."""
    attrs = ref.get("attributes") or {}
    out: list[str] = []
    for k in _REF_DATE_ATTR_KEYS:
        v = attrs.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def _gold_ref_searchable_from_instruction(
    instruction: str, ref: dict,
) -> tuple[bool, list[str]]:
    """Return (any_attr_hits, list_of_attempted_values).

    True when at least one searchable attribute value of `ref` is loose-
    matched by `instruction`. False means the instruction has no token
    overlap with any searchable attribute of the gold ref — agent cannot
    surface it via search, oracle will fail.

    When the ref has *no* searchable attributes at all (e.g. a calendar
    event identified only by title/start_time, with no location/region),
    returns (True, []) — caller cannot decide; treat as pass to avoid
    false alarms.
    """
    candidates = _ref_searchable_values(ref)
    if not candidates:
        return True, []
    for _, v in candidates:
        if _instruction_loose_hits(instruction, v):
            return True, []
    # No attribute matched — surface every attempted value for diagnostics.
    return False, [f"{k}={v}" for k, v in candidates]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_type_ok(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name in ("date", "datetime"):
        return isinstance(value, str)
    if type_name == "array[string]":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    m = re.fullmatch(r"enum\[(.+)\]", type_name)
    if m:
        enums = [s.strip() for s in m.group(1).split("|")]
        return isinstance(value, str) and value in enums
    return True


# Compare-arg safe-type rule lives in `lib.compare_utils` (shared between
# datagen and evaluator) so both stages enforce the same predicate without
# evaluator needing to import from datagen.
from lib.compare_utils import safe_for_compare as _safe_for_compare  # noqa: E402


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def static_check(session: dict, rule: dict) -> list[str]:
    """Returns list of error strings (empty = pass).

    `session`: output of test_session_gen (post-stamping — has session_id,
        session_type, rule_id, rule_ref, domain).
    `rule`: source rule dict (from rules_qc.json).
    """
    errors: list[str] = []
    tmap = tool_map()

    # ── 1. required top-level fields ──────────────────────────────────────
    for f in ("session_id", "session_type", "domain", "rule_id", "instruction",
              "local_env", "labels", "rule_ref"):
        if f not in session:
            errors.append(f"missing:{f}")
    if errors:
        return errors

    if session["session_type"] != "test":
        errors.append(f"session_type_must_be_test:{session['session_type']}")

    # ── 2. domain / rule_id consistency ────────────────────────────────────
    if session["domain"] not in DOMAINS:
        errors.append(f"unknown_domain:{session['domain']}")
    if session["rule_id"] != rule.get("rule_id"):
        errors.append(f"rule_id_mismatch:{session['rule_id']}:{rule.get('rule_id')}")

    # ── 3. rule_ref shape ──────────────────────────────────────────────────
    rr = session["rule_ref"]
    if not isinstance(rr, dict):
        errors.append("rule_ref_not_object")
    else:
        for k in ("rule_id", "rule_text", "canonical_answer"):
            if not rr.get(k):
                errors.append(f"rule_ref_missing:{k}")
        if rr.get("rule_id") != rule.get("rule_id"):
            errors.append("rule_ref_id_mismatch")

    # ── 4. local_env ───────────────────────────────────────────────────────
    le = session["local_env"]
    if not isinstance(le, dict):
        errors.append("local_env_not_object")
        return errors

    whitelist = le.get("tools") or []
    if not isinstance(whitelist, list) or not whitelist:
        errors.append("whitelist_empty")
        return errors

    base_names = base_tool_names()
    dom_tool_names = {t["name"] for t in domain_tools(session["domain"])}
    for t in whitelist:
        if t not in tmap:
            errors.append(f"whitelist_unknown:{t}")
        elif t not in dom_tool_names and t not in base_names:
            errors.append(f"whitelist_cross_domain:{t}:{tmap[t].get('domain')}")

    refs = le.get("references") or []
    if not isinstance(refs, list):
        errors.append("references_not_list")
        refs = []

    supported_ref_types = ref_types_for_domain(session["domain"])
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
            errors.append(f"ref_duplicate_id:{rid}")
        ref_ids.add(rid)
        if rtype not in supported_ref_types:
            errors.append(f"ref_type_invalid:{rtype}:domain={session['domain']}")
        # Top-level keys outside {id,type,attributes} are silently dropped by
        # ATRDB._unpack_ref → the env never sees them. Past failure mode
        # (observed case): the LLM put name/location/tags at top level,
        # static_check passed, runtime hydrated trip_stops with
        # name=ref_id / location=None / tags=[] and search_trip_stops
        # returned 2/6 candidates instead of 6. Reject on sight.
        extras = sorted(k for k in r.keys() if k not in ("id", "type", "attributes"))
        if extras:
            errors.append(f"ref_{rid}_top_level_extras:{extras}")
        if not isinstance(attrs, dict):
            errors.append(f"ref_{i}_attributes_not_object")
            continue
        # Evaluative attributes — reject on sight.
        for k in attrs.keys():
            if k in _EVALUATIVE_ATTRS:
                errors.append(f"ref_{rid}_evaluative_attr:{k}")

    # ── 5. labels.task_success.required_actions ────────────────────────────
    labels = session.get("labels") or {}
    ts = labels.get("task_success") or {}
    required = ts.get("required_actions") or []
    if not isinstance(required, list) or not required:
        errors.append("required_actions_empty")
        return errors

    is_permission = rule_is_permission(rule)
    # Flat single-step schema: every rule has one action_step → one
    # RequiredAction. Confirm rules' gold targets `get_user_confirmation`;
    # mutate rules' gold targets the chosen mutate tool directly.
    if len(required) != 1:
        errors.append(f"required_actions_must_be_len_1:got={len(required)}")

    # Decision-step tool must not be exploration-only (search/list/lookup/...).
    # Permission rules (Shape B) target `get_user_confirmation`, a base-domain
    # tool — exempt from the exploration prefix guard (base tools aren't
    # "exploration" in the search/list sense, and `get_` would otherwise
    # falsely trigger).
    if required and not is_permission:
        decision_tool = (required[0] or {}).get("tool", "") or ""
        if decision_tool.startswith(_EXPLORATION_TOOL_PREFIXES):
            errors.append(
                f"decision_step_is_exploration_tool:tool={decision_tool} — "
                "gold decision should be a final-action tool, "
                "not an exploration/read-only tool"
            )

    # per-action checks
    all_compare_args: set[str] = set()
    _compare_arg_types: dict[str, str] = {}  # arg_name → type_str (for check_type consistency)
    for idx, step in enumerate(required):
        if not isinstance(step, dict):
            errors.append(f"action_{idx}_not_object")
            continue
        tname = step.get("tool")
        args = step.get("arguments") or {}
        compare = step.get("compare_args")

        if tname not in tmap:
            errors.append(f"action_{idx}_unknown_tool:{tname}")
            continue
        if tname not in whitelist:
            errors.append(f"action_{idx}_tool_not_in_whitelist:{tname}")

        # arg types match tool signature
        sig = {p["name"]: p for p in tmap[tname]["parameters"]}
        for k, v in args.items():
            if k not in sig:
                errors.append(f"action_{idx}_arg_unknown:{tname}.{k}")
                continue
            if v is not None and not _value_type_ok(v, sig[k]["type"]):
                errors.append(
                    f"action_{idx}_arg_type:{tname}.{k}:{sig[k]['type']}"
                )

        # Gold ids must be present in references (or auto-preseed / search-type).
        # Applies to ALL steps, not just idx==0: chain decision steps (idx≥1) can
        # also reference fabricated ids that the oracle trace can't discover.
        # Secondary entity IDs (order_id, trip_id, etc.) are skipped here —
        # check 11 validates them against reference attributes with a specific
        # error message.
        is_search = tname in _SEARCH_TYPE_TOOLS
        for k, v in args.items():
            if isinstance(v, str) and k.endswith("_id"):
                if k in _SECONDARY_ENTITY_PARAMS:
                    continue  # handled by check 11
                if v not in ref_ids and v not in _AUTO_PRESEED_IDS and not is_search:
                    errors.append(f"action_{idx}_id_missing_from_refs:{tname}.{k}={v}")
            elif isinstance(v, list) and k.endswith("_ids"):
                for gid in v:
                    if not isinstance(gid, str):
                        continue
                    if gid not in ref_ids and gid not in _AUTO_PRESEED_IDS and not is_search:
                        errors.append(f"action_{idx}_id_missing_from_refs:{tname}.{k}={gid}")

        if compare is not None:
            if not isinstance(compare, list):
                errors.append(f"action_{idx}_compare_args_not_list")
            else:
                all_compare_args.update(compare)
                for c in compare:
                    if c not in args:
                        errors.append(f"action_{idx}_compare_arg_not_in_args:{c}")
                    elif c in sig:
                        _compare_arg_types[c] = sig[c]["type"]
                        # compare_args may only carry safe-typed slots —
                        # *_id / *_ids identifiers (compared as opaque
                        # strings against a closed pool), enum values, or
                        # booleans. Anything else (datetime, integer,
                        # number, free-text string, non-id array) makes
                        # gold == actual exact-match brittle: timestamps
                        # paraphrase, integers JSON-serialize as strings,
                        # free text never matches verbatim.
                        ptype = sig[c]["type"]
                        if not _safe_for_compare(c, ptype):
                            errors.append(
                                f"action_{idx}_compare_arg_unsafe_type:{tname}.{c}:{ptype}"
                            )

    # ── 5b. check_type ↔ required_action.compare_args consistency ─
    # Dispatched by check_type (4 flat branches). action_step is the
    # ground truth; compare_args must match what derive_required_actions
    # would produce for that check_type.
    rule_step = rule.get("action_step") or {}
    rule_check_type = rule.get("check_type")
    if isinstance(rule_step, dict) and required:
        param = rule_step.get("param")
        cmp = required[0].get("compare_args")
        cmp_set = set(cmp) if isinstance(cmp, list) else set()

        if rule_check_type == "confirm":
            valid_sets = ({"target_tool"}, {"target_tool", "target_params"})
            if cmp_set not in valid_sets:
                errors.append(
                    f"confirm_action_compare_args_must_be_one_of_"
                    f"[['target_tool'], ['target_tool','target_params']]:"
                    f"got={sorted(cmp_set)}"
                )
        elif rule_check_type in ("param_id", "param_enum"):
            if cmp_set != {param}:
                errors.append(
                    f"action_compare_args_must_match_rule_param:expected={[param]}:got={sorted(cmp_set)}"
                )
        elif rule_check_type == "tool_identity":
            if cmp is not None:
                errors.append("action_tool_identity_must_have_null_compare_args")

    # ── 7. instruction checks ──────────────────────────────────────────────
    instruction = (session.get("instruction") or "").strip()
    if not instruction:
        errors.append("instruction_empty")
    else:
        il = instruction.lower()
        # Leaky enum values (literal)
        for step in required:
            for v in (step.get("arguments") or {}).values():
                if isinstance(v, str) and v.lower() in _LEAKY_ENUM_VALUES and v.lower() in il:
                    errors.append(f"instruction_leaks_gold_value:{v}")
                if isinstance(v, str) and v in ref_ids and v in instruction:
                    errors.append(f"instruction_leaks_ref_id:{v}")
                # array[string] params (e.g. *_ids): each element is a ref id
                if isinstance(v, list):
                    for gid in v:
                        if isinstance(gid, str) and gid in ref_ids and gid in instruction:
                            errors.append(f"instruction_leaks_ref_id:{gid}")
                # Natural-language synonyms — these are the actual leak
                # vectors in practice since gold enum values rarely appear
                # verbatim in instructions. Case-insensitive substring match.
                if isinstance(v, str):
                    syns = _LEAKY_ENUM_SYNONYMS.get(v.lower())
                    if syns:
                        for syn in syns:
                            if syn and syn.lower() in il:
                                errors.append(
                                    f"instruction_leaks_gold_synonym:{v}→'{syn}'"
                                )
                                break

        # tool names must not appear verbatim (agent should derive from task, not name)
        for step in required:
            gt = step.get("tool", "")
            if gt and gt.lower() in il:
                errors.append(f"instruction_leaks_gold_tool:{gt}")

    # ── 8. check_type ↔ structure invariants ──────────────────────────────
    check_type = rule.get("check_type")

    # Tool-identity whitelist decoy requirement relaxed — under the
    # whole-domain exposure policy, the domain always has ≥2 non-base tools,
    # so this check is trivially satisfied. Kept as a sanity guard against
    # truncated whitelists (e.g. if someone hand-writes a session with fewer).
    if check_type == "tool_identity":
        non_base = [t for t in whitelist if t in tmap and tmap[t].get("domain") != "base"]
        if len(non_base) < 2:
            errors.append("tool_identity_whitelist_needs_decoy")

    # param_* check types require at least one action with compare_args
    has_param_check = check_type in ("param_id", "param_enum")
    if has_param_check and not all_compare_args:
        errors.append("param_check_but_no_compare_args")

    # tool_identity (mutate) rules: compare_args MUST be null. Confirm rules
    # have their own compare_args validation in 5b.
    if check_type == "tool_identity":
        for idx, step in enumerate(required):
            if step.get("compare_args") is not None:
                errors.append(f"tool_identity_only_but_action_{idx}_has_compare_args")

    # param_id rule: references must have ≥3 objects of the target ref_type
    # (heuristic — ensures decoys exist). Under the single-axis schema,
    # every rule's gold is one tool call; param_id rules always need a
    # 1-gold + N-decoy candidate pool.
    if check_type == "param_id" and len(refs) < 3:
        errors.append(f"param_id_needs_at_least_3_refs:got={len(refs)}")

    # check_type ↔ compare_args type consistency (mutate rules only)
    if has_param_check and _compare_arg_types:
        covered = {"param_id": False, "param_enum": False}
        for aname, ptype in _compare_arg_types.items():
            if aname.endswith("_id") or aname.endswith("_ids"):
                covered["param_id"] = True
            if ptype.startswith("enum["):
                covered["param_enum"] = True
        if not covered.get(check_type):
            errors.append(
                f"{check_type}_check_type_but_no_matching_compare_arg"
            )

    # ── 9. instruction-ref keyword consistency ────────────────────────────
    # Every gold-referenced ref id must be surfaceable by some search/list
    # query derivable from instruction tokens. Concretely, the ref must
    # carry at least one searchable attribute (location / name / sender /
    # cuisine / service_type / region / tags / ...) whose value loose-
    # matches at least one instruction token.
    #
    # Why this check matters: when the instruction and the ref use
    # different languages or wordings for the same place (e.g. a localized
    # instruction term vs. an English-only ref location), the agent's
    # natural search query returns 0 results, so it gives up and oracle
    # fails. Time-mismatch cases follow the same pattern on date attributes;
    # we don't enforce date alignment here (it needs instruction
    # time-parsing) — keyword alignment is the more general check.
    #
    # This check intentionally PASSES when the ref has no searchable
    # attributes (e.g. calendar_event with only title/start_time/
    # participants and no location field) — we can't decide for those,
    # and false alarms are worse than misses at static layer.
    refs_by_id = {r.get("id"): r for r in refs if isinstance(r, dict) and r.get("id")}
    gold_id_origins: dict[str, list[str]] = {}  # ref_id → [step_idx:tool, ...]
    for idx, step in enumerate(required):
        if not isinstance(step, dict):
            continue
        tname = step.get("tool", "")
        for k, v in (step.get("arguments") or {}).items():
            if isinstance(v, str) and k.endswith("_id"):
                if v in refs_by_id:
                    gold_id_origins.setdefault(v, []).append(f"{idx}:{tname}")
            elif isinstance(v, list) and k.endswith("_ids"):
                for gid in v:
                    if isinstance(gid, str) and gid in refs_by_id:
                        gold_id_origins.setdefault(gid, []).append(f"{idx}:{tname}")
    for gid, origins in gold_id_origins.items():
        ref = refs_by_id.get(gid)
        if not ref:
            continue
        ok, attempted = _gold_ref_searchable_from_instruction(instruction, ref)
        if not ok:
            origin_str = ",".join(origins)
            errs_summary = "|".join(attempted[:4])
            errors.append(
                f"instruction_keyword_misses_gold_ref:{gid} "
                f"(used_at=[{origin_str}], no instruction token loose-matches "
                f"any searchable attr; tried: {errs_summary}). "
                "Add a token from one of these attrs to instruction "
                "(or ensure the ref's location/name/etc. uses the same "
                "language/wording the instruction uses)."
            )

    # ── 10a. relative-time phrases forbidden in instruction ──────────────
    # No-time-axis: runtime hides the current date, so the agent cannot
    # resolve relative phrases like "next Thursday".
    for pat in _RELATIVE_TIME_PATTERNS:
        m = pat.search(instruction)
        if m:
            errors.append(
                f"instruction_contains_relative_time:'{m.group(0)}' — "
                "use absolute date YYYY-MM-DD or omit time entirely "
                "(MVP-no-time-axis: agent can't resolve relative phrases)"
            )
            break  # one match is enough; don't spam

    # ── 10b. absolute-date consistency: instruction date ↔ gold ref date ─
    # When instruction contains a YYYY-MM-DD token, every gold-referenced
    # ref with date attributes must contain that date as substring in at
    # least one date attr. Catches "May 18 + 2025-05-18 ref" type
    # year/month mismatches.
    abs_dates_in_instr = _ABS_DATE_RE.findall(instruction)
    if abs_dates_in_instr:
        for gid in gold_id_origins:
            ref = refs_by_id.get(gid)
            if not ref:
                continue
            ref_dates = _ref_date_values(ref)
            if not ref_dates:
                # Ref carries no date attrs — instruction's date is either
                # decorative (location event with no date field) or is
                # consumed outside ref-pool semantics. Skip.
                continue
            ref_blob = " | ".join(ref_dates)
            if not any(d in ref_blob for d in abs_dates_in_instr):
                errors.append(
                    f"instruction_date_misses_gold_ref:{gid} "
                    f"(instr_dates={abs_dates_in_instr}, "
                    f"ref_dates={ref_dates}). Align ref's date attr to "
                    "instruction's absolute date, or drop the date from "
                    "instruction so agent picks by ref's date_time field."
                )

    # ── 11. Secondary entity ID coverage (env hydration) ────────────────
    # Tools like modify_order / replan_trip / cancel_service_appointment
    # operate on secondary entities (orders, trips, appointments) that the
    # env creates during hydration from reference attributes.  When a gold
    # action targets such an entity, at least one reference of the
    # corresponding type must carry the entity ID in its attributes.
    for idx, step in enumerate(required):
        if not isinstance(step, dict):
            continue
        tname = step.get("tool", "")
        entity_map = _SECONDARY_ENTITY_ID_MAP.get(tname)
        if not entity_map:
            continue
        args = step.get("arguments") or {}
        for id_param, required_ref_type in entity_map.items():
            id_val = args.get(id_param)
            if not isinstance(id_val, str) or not id_val:
                continue
            found = False
            for r in refs:
                if not isinstance(r, dict):
                    continue
                if r.get("type") != required_ref_type:
                    continue
                attrs = r.get("attributes") or {}
                if attrs.get(id_param) == id_val:
                    found = True
                    break
            if not found:
                errors.append(
                    f"action_{idx}_secondary_entity_missing:{tname}.{id_param}={id_val} "
                    f"(no ref of type '{required_ref_type}' carries "
                    f"{id_param}={id_val!r} in its attributes — "
                    f"env cannot hydrate this entity. "
                    f"Add {id_param} to a {required_ref_type} ref's attributes)"
                )

    # ── 12. Runtime executability dry-run (build env, execute gold) ──────
    if not errors:
        errors.extend(_executability_dry_run(session, rule))

    return sorted(set(errors))


# ---------------------------------------------------------------------------
# check 12 executability dry-run
# ---------------------------------------------------------------------------


def _executability_dry_run(session: dict, rule: dict) -> list[str]:
    """Build env from session.local_env.references + a dummy persona,
    then verify DB-level entity existence for secondary-entity actions.

    TS required_actions only carry gold-critical arguments (not all
    params the tool needs), so we cannot execute tools directly. Instead:
      1. Build env — catches hydration/build failures
      2. Check secondary entity IDs exist in the DB table

    Catches: hydration gaps, env build failures.

    Does NOT catch: discoverability (check 9), decoy quality (QC layer),
    argument shape errors (those are caught by check 1-check 11).
    """
    from runner.environment.base import PersonaProfile
    from runner.environment.domains import build_env_for_session
    from runner.schemas import TestSession as _TS

    try:
        session_obj = _TS(**session)
    except Exception as e:
        return [f"dry_run_session_build_failed: {e}"]

    persona = PersonaProfile(
        persona_id="__dryrun__",
        default_shipping_address="10 Main St, New York, NY 10001",
        default_payment_method="CARD_VAL001",
        default_contact="dryrun@example.com",
    )
    try:
        env = build_env_for_session(session_task=session_obj, persona=persona)
    except Exception as e:
        return [f"dry_run_env_build_failed: {e}"]

    errors: list[str] = []
    db = env.toolkit.db
    required = (
        session.get("labels", {})
        .get("task_success", {})
        .get("required_actions", [])
    ) or []
    for i, action in enumerate(required):
        tool = action.get("tool", "")
        args = action.get("arguments") or {}
        checks = _SECONDARY_ENTITY_CHECKS.get(tool, [])
        for id_param, table_name in checks:
            entity_id = args.get(id_param)
            if not entity_id:
                continue
            table = getattr(db, table_name, None)
            if table is None:
                errors.append(
                    f"dry_run_action_{i}_no_table: {tool} needs "
                    f"db.{table_name} but domain has no such table"
                )
            elif entity_id not in table:
                errors.append(
                    f"dry_run_action_{i}_entity_missing: {tool}.{id_param}="
                    f"{entity_id!r} not in db.{table_name} "
                    f"(hydration gap — ref missing {id_param} attribute)"
                )
    return errors


# Secondary-entity checks: tool → [(id_param, db_table_name)]
_SECONDARY_ENTITY_CHECKS: dict[str, list[tuple[str, str]]] = {
    "modify_order": [("order_id", "orders")],
    "cancel_order": [("order_id", "orders")],
    "return_order": [("order_id", "orders")],
    "track_order": [("order_id", "orders")],
    "replan_trip": [("trip_id", "trips")],
    "track_trip_updates": [("trip_id", "trips")],
    "modify_restaurant_reservation": [("reservation_id", "restaurant_reservations")],
    "cancel_restaurant_reservation": [("reservation_id", "restaurant_reservations")],
    "modify_event_ticket": [("booking_id", "event_bookings")],
    "cancel_event_ticket": [("booking_id", "event_bookings")],
    "modify_service_appointment": [("appointment_id", "service_appointments")],
    "cancel_service_appointment": [("appointment_id", "service_appointments")],
    "track_reservation_updates": [
        ("reservation_id", "restaurant_reservations"),
        ("booking_id", "event_bookings"),
        ("appointment_id", "service_appointments"),
    ],
}
