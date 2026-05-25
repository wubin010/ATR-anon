# Learning-Session Fill

The skeleton stage produced a thin skeleton (`domain` + `theme_one_line`). Fill in one complete learning session from that skeleton.

## Core constraints

### Write daily interactions only — do not reveal preferences

You hold no long-term rule pool. Write only factual scenarios (one specific thing this persona is handing off to the agent right now). Do not hint, in instruction / task_params / references, at "this person generally prefers X" or any cross-task decision direction.

### Lazy-user contract

- **`reason_for_call`** is what the user holds in mind as the overall reason for contacting the agent — only `user_sim` reads it; the agent never sees it. user_sim uses it to generate the opening message naturally (vague, direction-only).

  `reason_for_call` MUST be vague: convey direction / occasion / mood, but must NOT contain any specific `task_params` value (cuisine, category, location, brand, person name, event title, project name, file / folder name, document type, or any concrete identifier). Generic container words any user might say upfront ("inbox", "calendar", "tracker", "folder", "trip") are fine; specific content words that name *this* user's target are not.

- **`task_params`** is the user's full mental list of specific needs, a flat dict `{field_name: concrete_value}`. Value types: numbers, dates, strings, ids, enum literals, lists — whatever the field needs. Do NOT add desc / rationale fields; user_sim will paraphrase at runtime.

- Runtime rule for user_sim: when the agent asks about a `task_params` field, paraphrase the value; otherwise reply "I don't know / whatever you think". Repeat the answer if the same field is asked twice.

### Closed-loop traceability (HARD invariants — every `task_params` key needs a runtime destination)

Every key falls into exactly one of these destinations:

| Destination | Key shape | Value constraint |
|---|---|---|
| A. Direct tool argument | matches a parameter name on a tool the session expects to use | type matches; if enum, value ∈ allowed set |
| B. Search filter against refs | matches a `local_env.references[*].attributes` key on the matching ref type | value MUST appear lexically in at least one reference's attribute value |
| C. Reference id selector | ends with `_id` or `_ids` | value(s) MUST exist in `local_env.references[*].id` |
| D. Drafting points | ends with `_points` / `_keywords` / `_bullets` | `list[str]`, each item ≤ ~80 chars; NOT a pre-authored body |

**Forbidden in `task_params`:**

- **E. Runtime autofill fields** — `contact_info` / `payment_method` / `shipping_address` / `traveler_info` / `guest_info`. These are auto-injected from PersonaProfile; including them is redundant and will be rejected.
- **F. Server-generated derived IDs** — `order_id` / `trip_id` / `booking_id` / `appointment_id` / `reservation_id`. LS is first-contact; the user does NOT know server IDs (`ord_xxx`). (TS hydrates derived entities from primary ref attributes; LS does not — derived IDs in LS refs are also forbidden.) Use a `<noun>_description` field instead (e.g. `order_description: "the recent bird-seed order"`); the agent resolves the actual id via `track_*` / `search_*` at runtime.
- Algorithmic / procedural instructions encoded as values (`selection_rule: "choose the first conflict-free option..."`). Values are *facts*, not *agent algorithms*.
- Output-control booleans (`include_daily_breakdown: true`). Agent output knobs, not user-side facts.

### References pool must support `gold_trajectory`

The trajectory you write must be physically executable against your refs. If the trajectory contains a search step, refs must include ≥1 of the corresponding type, or the agent stalls.

### Granularity match

Any string-typed `task_params` value used as a search filter (location / cuisine / category / region) must appear lexically (token-level overlap) in at least one ref attribute. If the user's mental notion is at a different granularity than refs, use the granularity that lexically overlaps refs.

Counter-example. `task_params.location = "Upper Manhattan"` while refs say `location: "Washington Heights"` — no token overlap, so the env's substring filter returns 0.

Example. `task_params.location = "Washington Heights"` matching `refs[*].attributes.location = "Washington Heights"`.

---

## Inputs

### Skeleton (the skeleton stage has set this; do NOT change domain or invent themes)

```json
{{SKELETON}}
```

### Persona structured data (background, for grounding)

{{STRUCTURED_PERSONA}}

### Domain `{{DOMAIN}}` available tool specs

```json
{{DOMAIN_TOOLS}}
```

### Domain `{{DOMAIN}}` reference object schema

{{REF_SCHEMA}}

---

## Output JSON schema

```json
{
  "reason_for_call": "<≤10 word verb+noun phrase>",
  "task_params": {"<field_1>": <concrete value>, ...},
  "local_env": {
    "references": [
      {"id": "<id with short suffix>", "type": "<valid ref_type>", "attributes": {...}},
      ...
    ]
  },
  "gold_trajectory": [
    {"tool": "<tool name from this domain>", "arguments": {...}},
    ...
  ]
}
```

> `local_env.tools` and `expected_tools`: you do not output these — the caller fills `tools` with the whole-domain list and derives `expected_tools` from your `gold_trajectory`.

## Field requirements

### `reason_for_call`

Ultra-abstract phrase, ≤ 10 English words, expressing only "verb + general object". `user_sim` reads it and paraphrases it as the opening message; everything specific stays in `task_params` and is disclosed only when the agent asks.

Strictly forbidden in `reason_for_call`:

- Any `task_params` value, verbatim or as a synonym (cuisine / category / location / brand / event title).
- Any time-of-day / day-of-week / specific occasion / mood adjective ("this weekend", "before today's meeting", "to celebrate").
- Any "head start" that lets the agent skip asking.

Example. "Help me with a dinner reservation."

Counter-example. "Help me book a sushi place in SoHo for two later this week." (Leaks cuisine, location, party size, and date hint.)

Length check: if the draft has more than 10 words, strip everything except verb + general noun phrase.

### `task_params` (3–10 fields, flat key → value)

- Include ALL the information the agent needs to complete the task.
- Values are concrete (numbers, dates, place names, headcounts, ids, enum values) — no placeholders.
- Every field must be answerable by user_sim when asked. Since the instruction reveals nothing, every field is obtained by the agent asking or searching.
- Every (key, value) satisfies one of destinations A--D in the traceability table above.
- Do not stuff agent output text into a value: no full body for `reply_body` / `document_content` — use a `*_points` drafting list.
- **Executability**: think clearly about the last action tool the agent calls (`book_` / `place_` / `send_` / `create_` / `cancel_` / ...). Its required parameters must come from a value in `task_params`, an id from `local_env.references`, or a runtime auto-injected identity field (`contact_info` / `traveler_info` / `guest_info` / `shipping_address` / `payment_method` — these MUST NOT appear in task_params).

### `gold_trajectory` (oracle solution path — 2–5 steps)

The sequence of tool calls an agent would make if it knew all `task_params` upfront (no asking). Each step is `{"tool": <name>, "arguments": <dict>}`. The caller derives unique tool names from this trajectory.

Hard requirements:

- Each step's `tool` is in this domain's tool list (`{{DOMAIN_TOOLS}}`).
- **Every concrete argument value** (required and optional, search filters and write payloads) must come from one of:
  1. A value in `task_params` (verbatim).
  2. An id from `local_env.references[*].id`.
  3. A placeholder `<authored from <field>[, <field>...]>` for long-text drafting args (every referenced `<field>` must exist in `task_params`).
  4. A runtime auto-injected identity field, OMITTED from arguments (env auto-fills).
  5. A recoverable broad-list default for narrowing filters the agent could safely omit: `search_messages.folder ∈ {inbox, sent, drafts}` (omitting returns all non-archived messages); `list_events.calendar` (omitting returns all calendars). NOT recoverable: `search_messages.folder = "archive"` (archived messages are hidden by default — this must come from task_params).
  6. An agent-side control flag (`sort_by`, `limit`, `language` defaults, output-shaping booleans, `field` / `update_action` selectors). These are how the agent invokes the tool, not user-side facts, and need not appear in task_params.
- Anything else (a `doc_type`, `cuisine`, `location`, `sender`, `priority`, `title`, `start_time`, `participants` invented only in `gold_trajectory`) is oracle-only knowledge user_sim cannot disclose — fail.
- Every `search_*` / `list_*` step must be answerable by your refs (≥1 ref of the matching type satisfying the step's filters).
- Every id used as an argument must exist in `local_env.references` with the right type.
- The last step is typically the action tool. Trajectory length 2–5 (search → optional compare → action).

For long-text drafting args (e.g. `body` on `draft_message`, `content` on `create_document`), use a placeholder like `"<authored from reply_points>"` — do not compose the full text.

**Examples** (replace `<task_params.X>` with the literal values you wrote):

```json
// reservation
[
  {"tool": "search_restaurants", "arguments": {"cuisine": "<task_params.cuisine>", "location": "<task_params.location>", "sort_by": "rating"}},
  {"tool": "book_restaurant",   "arguments": {"restaurant_id": "<gold ref id>", "date_time": "<task_params.date_time>", "party_size": 2}}
]

// communication
[
  {"tool": "search_messages",      "arguments": {"folder": "inbox", "sender": "<task_params.target_sender>"}},
  {"tool": "set_message_priority", "arguments": {"message_ids": ["<msg id>"], "priority": "high"}},
  {"tool": "draft_message",        "arguments": {"thread_id": "<thread id>", "language": "en", "body": "<authored from reply_points>"}}
]
```

### `local_env.references` (3–6 entries)

- Each ref: `{id, type, attributes}`. `type` must be valid in `{{REF_SCHEMA}}` — do not invent. `attributes` follow the schema contract. `id` follows the schema's recommended prefix (`rst_`, `prod_`, `msg_`) with a numeric suffix (e.g. `rst_momoya_01`).
- Cover every search-style tool in `gold_trajectory`: if a step is `search_restaurants`, refs must include ≥1 `restaurant`; same for `search_destinations` → `destination`, `list_events` → `calendar_event`.
- Diverse and natural distribution — avoid 4 near-identical items.
- References hold only pre-existing candidate objects available before the agent starts. Do not put result objects (orders already placed, bookings already made) in references; use a descriptive field like `order_description: "the recent art-supply order"` and let the agent fetch via `track_*` / `search_*`.
- Derived IDs in ref attributes are LS-forbidden. REF_SCHEMA's `derived:` sections describe TS-only env hydration; LS is interactive first-contact with no hydration, so do NOT put `order_id` / `trip_id` / `booking_id` / `reservation_id` / `appointment_id` in any ref's attributes.

---

## Domain-specific atomic constraints

### scheduling — `title` + `start_time` + `end_time` must come from the SAME calendar_event ref

If `task_params` includes any of `title` / `start_time` / `end_time` for an existing event, they must jointly match a single `calendar_event` ref's attributes. Treat the triplet as one atomic write — change one, sync the other two. Picking the title from one ref and the time from another (or made-up time) is the most common scheduling-domain failure.

### workspace — reference doc_type uses a dedicated `task_params` field

When `gold_trajectory.search_documents` uses a `doc_type` different from `task_params.doc_type` (the agent looks at an existing template / reference before creating the user's actual document), declare the search type under one of: `source_doc_type` / `template_doc_type` / `reference_doc_type`. If multiple reference types are consulted, use a distinct field for each.

Example. `task_params: {doc_type: "note", source_doc_type: "template", ...}` → gold: `search_documents(doc_type="template")` then `create_document(doc_type="note")`.

Counter-example. `task_params: {doc_type: "note", ...}` → gold: `search_documents(doc_type="template")` — no `task_params` field carries the reference type, so user_sim cannot answer if asked.

---

JSON only — no markdown code fence, no other text.
