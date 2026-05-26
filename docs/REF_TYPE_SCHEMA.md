# Reference Type Schema

> Each reference under `session_task.local_env.references` is an
> `{id, type, attributes}` triple (see `runner/schemas.py:ReferenceObject`).
> At runtime, each domain DB calls `from_references` to lift these entries
> into typed model objects. This document lists the attribute contract for
> each legal `type` value — fill these fields per the
> tables when authoring a session_task, so the runtime can correctly load
> the references into the DB for tools to query.
>
> Writing the wrong `type` (e.g. `"item"` instead of `"product"`) raises
> a runtime **WARNING** but does **not** stop execution — the reference
> is silently dropped, leaving the env empty and the payoff evaluation
> failing. Don't hit this trap.

---

## 1. Overview: Domain × legal ref_type

| Domain | Legal `type` values | Recommended id prefix |
|---|---|---|
| commerce | `product`, `subscription` | `prod_*`, `sub_*` |
| reservation | `restaurant`, `event`, `service_provider` | `rst_*`, `evt_*`, `prv_*` |
| travel | `destination`, `flight_offer`, `hotel_offer`, `transport_offer`, `trip_stop` | `dst_*`, `flt_*`, `hot_*`, `gnd_*`, `stp_*` |
| communication | `message`, `priority_bucket`, `folder`, `label` | `msg_*`, `pb_*`, `fld_*`, `lbl_*` |
| scheduling | `calendar_event` (preferred) / `event` (alias) | `evt_*` |
| workspace | `file`, `document`, `tracker`, `folder` | `fil_*`, `doc_*`, `trk_*`, `fld_*` |

> ⚠️ **`folder` ref_type is shared between communication and workspace** (v0.12) —
> each domain's DB has an independent folder dict, but the schema contract
> and id prefix are identical. `priority_bucket` / `label` exist only in
> communication.

> ⚠️ **scheduling.event and reservation.event are different things**
> (the former is a calendar event, the latter is a ticketed activity);
> they don't collide because they live in different domains. But a single
> session is in exactly one domain — the two never mix. For scheduling,
> prefer writing `calendar_event` to avoid ambiguity.

---

## 2. Detailed contracts (per domain)

### 2.1 commerce

#### `type: "product"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | display name |
| `price` | float | ✓ | unit price (USD) |
| `category` | str | ✓ | category, filtered by `search_products.category` (episode layer may tighten to enum) |
| `rating` | float? | ✓ | rating, used by `search_products(sort_by="rating")` |
| any other field | any | ✗ (lands in `attributes` dict) | e.g. `hardiness`, `brand`, `color` — accessed via `p.attributes.get(...)` |

**Consumed by**: `search_products` (name/category/budget/rating filter+sort), `compare_products` (id lookup in DB), `place_order` (product_id must exist in DB).

**Pre-existing orders** (test sessions only): when a test session's gold action targets an existing order (modify/cancel/return/track), add these fields so the env can hydrate an `Order` object:

| Field | Type | Notes |
|---|---|---|
| `order_id` | str? | triggers Order hydration; `modify_order`/`cancel_order`/`return_order`/`track_order` use this as `order_id` |
| `order_status` or `delivery_status` | str? | order status (confirmed / delivered / shipped / etc.); defaults to `"confirmed"` |

```json
{
  "id": "prod_lavender_01",
  "type": "product",
  "attributes": {
    "name": "English Lavender Plant",
    "price": 35,
    "category": "garden_plants",
    "rating": 4.3,
    "hardiness": "zone 5-9"
  }
}
```

Product with a pre-existing order:
```json
{
  "id": "prod_cedar_01",
  "type": "product",
  "attributes": {
    "name": "Cedar Glow Candle",
    "price": 29.99,
    "category": "candles",
    "order_id": "ORD-74821",
    "order_status": "delivered"
  }
}
```

#### `type: "subscription"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | |
| `plan` | str | ✓ | plan name (basic / pro / ...) |
| `status` | str | ✓ | active / paused / cancelled |
| `pause_until` | str? | ✓ | ISO date |
| `price_per_period` | float? | ✓ | |
| Others | any | ✗ | — |

**Anomaly markers for `review_recurring_charges(focus=...)`** (placed inside attributes):
- `price_changed: true` or `previous_price_per_period: <num>` → matched by focus=price_change
- `usage_anomaly: true` or `usage_notes: "<str>"` → focus=usage
- `plan_changed: true` or `tier_change: "<str>"` → focus=tier

**Consumed by**: `review_recurring_charges` (focus filter), `pause/resume/change/cancel_subscription` (id lookup in DB).

```json
{
  "id": "sub_streamflix_01",
  "type": "subscription",
  "attributes": {
    "name": "StreamFlix Premium",
    "plan": "pro",
    "status": "active",
    "price_per_period": 15.99,
    "price_changed": true,
    "previous_price_per_period": 12.99
  }
}
```

---

### 2.2 reservation

#### `type: "restaurant"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | |
| `cuisine` | str | ✓ | path-2 string; episode layer may tighten to enum |
| `location` | str | ✓ | city / area |
| `rating` | float? | ✓ | |
| `price_range` | str? | ✓ | `$` / `$$` / `$$$` / `$$$$` |
| `seating_style` | str? | ✓ | added in v0.12. Neutral factual descriptor ("quiet dining room" / "bar counter" / "outdoor patio" / "shared tables" / "booths and separated dining rooms"). **Gold landing for dining-environment rules**: the payoff candidate pool mixes seating_style values, and the gold restaurant_id points to the one matching the rule's direction. |
| Others | any | ✗ | |

**Consumed by**: `search_restaurants` (cuisine/location/rating/sort_by), `book_restaurant` (the `seating_preference` param was removed in v0.12; preference is now expressed via the chosen restaurant_id).

**Pre-existing reservations** (test sessions only): when the gold action targets an existing reservation (modify/cancel), add these fields:

| Field | Type | Notes |
|---|---|---|
| `reservation_id` | str? | triggers RestaurantReservation hydration |
| `reservation_status` | str? | defaults to `"confirmed"` |
| `reservation_date_time` | str? | ISO datetime of the reservation |
| `party_size` | int? | defaults to `1` |

#### `type: "event"` (reservation domain — ticketed activities)

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | |
| `event_type` | str | ✓ | path-2 string (exhibition/festival/talk/...) |
| `location` | str | ✓ | |
| `date_time` | str? | ✓ | ISO |
| `price` | float? | ✓ | |
| `event_tags` | list[str] | ✓ | path-2 string array, used by `search_events.event_tags` filter |
| Others | any | ✗ | |

**Consumed by**: `search_events` (event_type/event_tags/location filter), `book_event_ticket`.

**Pre-existing event bookings** (test sessions only): when the gold action targets an existing booking (modify/cancel), add these fields:

| Field | Type | Notes |
|---|---|---|
| `booking_id` | str? | triggers EventBooking hydration |
| `booking_status` | str? | defaults to `"confirmed"` |
| `ticket_type` | str? | defaults to `"standard"` |
| `ticket_count` | int? | defaults to `1` |

#### `type: "service_provider"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | |
| `service_type` | str | ✓ | path-2 string |
| `location` | str | ✓ | |
| `rating` | float? | ✓ | |
| `distance_km` | float? | ✓ | used by `sort_by="distance"` |
| Others | any | ✗ | |

**Consumed by**: `search_service_providers`, `book_service_appointment`.

**Pre-existing appointments** (test sessions only): when the gold action targets an existing appointment (modify/cancel), add these fields:

| Field | Type | Notes |
|---|---|---|
| `appointment_id` | str? | triggers ServiceAppointment hydration |
| `appointment_status` | str? | defaults to `"confirmed"` |
| `appointment_date_time` | str? | ISO datetime of the appointment |

---

### 2.3 travel

#### `type: "destination"`

| Field | first-class | Notes |
|---|---|---|
| `name`, `region`, `tags` (list[str]) | ✓ | `search_destinations.destination_constraints` matches against region/tags/name |

#### `type: "flight_offer"`

| Field | first-class | Notes |
|---|---|---|
| `origin`, `destination`, `departure_date`, `airline`, `price` | ✓ | `search_flights` filters on origin/destination, sorts by price via `sort_by` |

#### `type: "hotel_offer"`

| Field | first-class | Notes |
|---|---|---|
| `name`, `location`, `price_per_night` | ✓ | `search_hotels` filters on location, sorts on `price_per_night` via `sort_by` |

#### `type: "transport_offer"`

| Field | first-class | Notes |
|---|---|---|
| `origin`, `destination`, `mode` (train/bus/car_transfer), `departure_date`, `price` | ✓ | `search_ground_transport` |

#### `type: "trip_stop"`

| Field | first-class | Notes |
|---|---|---|
| `name`, `stop_type`, `location`, `tags` (list[str]) | ✓ | `search_trip_stops`: destination matches `name`/`location` as substring; `stop_type` matches strictly; `stop_tags` filters by intersection |

**Pre-existing trips** (test sessions only): when the gold action targets an existing trip (replan/track), trip_stop refs must carry a shared `trip_id`:

| Field | Type | Notes |
|---|---|---|
| `trip_id` | str? | shared across all stops in the same trip — triggers TripPlan hydration; all stops with the same `trip_id` are grouped into one trip's `selected_stop_ids` |

> **travel.bookings limitation**: The `bookings` table (`modify_flight_booking`, `modify_hotel_booking`, `cancel_flight_booking`, `cancel_hotel_booking`, `cancel_ground_transport_booking`) has no hydration source and no search tool to discover booking IDs. These 5 tools **cannot anchor a rule's gold action** in test sessions. They are used in learning sessions only, where the agent first calls `book_*` and then chains to modify/cancel. This is enforced by the binding-stage lint (TableSpec `discovery_tools=[]`).

---

### 2.4 communication

#### `type: "message"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `sender` | str | ✓ | substring match for `search_messages.sender` |
| `subject` | str | ✓ | |
| `body` | str | ✓ | |
| `thread_id` | str? | ✓ | |
| `folder` | str | ✓ | defaults to `"inbox"`; the folder name where this message currently lives |
| `labels` | list[str] | ✓ | appended to by `label_messages`; stores the label object's `name` (after dereference) |
| `priority` | str | ✓ | `"high"/"medium"/"normal"/"low"` (written back after dereferencing the bucket) |
| `archived` | bool | ✓ | flipped by `archive_messages` |
| `timestamp` | str? | ✓ | |
| Others | any | ✗ | |

**Consumed by**: `search_messages`, `set_message_priority` (replaces prioritize_messages in v0.12), `archive_messages`, `label_messages`, `draft_message`/`send_message`.

#### `type: "priority_bucket"` (added in v0.12)

| Field | Type | first-class | Notes |
|---|---|---|---|
| `level` | str | ✓ | `"high"` / `"medium"` / `"low"` / `"normal"` |

**Role**: target of `set_message_priority`'s `priority_bucket_id` parameter.

**Auto-preseeded** (new in v0.12): `CommunicationDB.from_references` **automatically preseeds** the 4 standard buckets — `pb_high` / `pb_medium` / `pb_low` / `pb_normal` (with levels high/medium/low/normal respectively). **Data authors do not need to write them in `local_env.references` by hand**. To override a level, write a ref with the same id explicitly.

**Rule gold landing**: for priority-related rules, the gold = a bucket_id pointing to a specific level. Example: comm_03 "down-prioritize health newsletters" → gold = `set_message_priority(priority_bucket_id="pb_low")`. These ids are stable across sessions — once memory captures the literal "pb_low", the next session can reuse it directly.

#### `type: "folder"` (added in v0.12, shared by communication + workspace)

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | display name / path string (`"archived_hr"` / `"/training"` / `"inbox_processed"`) |
| `purpose` | str? | ✓ | purpose descriptor (`"hr_materials"` / `"training_resources"` / `"client_deliverables"`) |

**Role**: target of archive / move tools:
- communication: `archive_messages(folder_id)`
- workspace: `move_files(folder_id)`, `archive_files(folder_id)`

**Rule gold landing**: for filing-related rules, the gold = a folder_id whose attributes match the rule. Example: workspace_03 "training files go to /training" → gold = `move_files(folder_id=<some id whose purpose="training_resources">)`.

```json
{"id":"fld_training_01","type":"folder","attributes":{"name":"/training","purpose":"training_resources"}}
{"id":"fld_project_q3","type":"folder","attributes":{"name":"/projects/q3","purpose":"project_deliverables"}}
```

#### `type: "label"` (added in v0.12)

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name` | str | ✓ | label name (`"seahawks"` / `"project_urgent"` / `"receipts"`) |
| `topic` | str? | ✓ | topic category (`"sports_team"` / `"work"` / `"finance"`) |

**Role**: target of `label_messages(label_ids=...)`. Each session env preseeds a label pool.

**Rule gold landing**: comm_02 "tag Seahawks messages separately" → gold = `label_messages(label_ids=[<some id whose topic="sports_team" and name contains seahawks>])`.

```json
{"id":"lbl_seahawks","type":"label","attributes":{"name":"seahawks","topic":"sports_team"}}
{"id":"lbl_receipts","type":"label","attributes":{"name":"receipts","topic":"finance"}}
```

---

### 2.5 scheduling

#### `type: "calendar_event"` (preferred) or `type: "event"` (alias)

| Field | Type | first-class | Notes |
|---|---|---|---|
| `title` (or alias `name`) | str | ✓ | |
| `start_time` | str | ✓ | ISO datetime |
| `end_time` | str | ✓ | ISO datetime |
| `participants` | list[str] | ✓ | list of emails / ids |
| `location` | str? | ✓ | |
| `calendar` | str | ✓ | defaults to `"default"` |
| `status` | str | ✓ | confirmed / declined / tentative / cancelled |
| Others | any | ✗ | |

**Consumed by**: `list_events` (calendar/participants filter), `check_conflicts` (time-range comparison), `reschedule_event`, `modify_event`, `respond_to_event_invite`, `cancel_event`, `set_reminder` (target_id must exist in this DB), `track_event_updates`, `find_alternative_slots`.

**⚠️ Note**: `set_reminder.target_id` must hit an id in this table — meaning that to test "set a reminder after creating an event", the env must either have a calendar_event preseeded, or the agent must first call `create_event` to produce an id and then call `set_reminder`.

---

### 2.6 workspace

#### `type: "file"`

| Field | Type | first-class | Notes |
|---|---|---|---|
| `name`, `file_type`, `location`, `owner`, `labels`, `archived` | ✓ | `list_files` filters by location prefix + file_type |
| `attributes.{project, modified_date, priority}` | ✗ | | `classify_files.scheme` aggregates labels by these fields |

**Consumed by**: `list_files`, `classify_files` (uses `attributes.project` / `modified_date` / `priority`), `move_files` (takes folder_id, v0.12), `archive_files` (takes folder_id, v0.12), `delete_files`.

#### `type: "folder"` (added in v0.12, contract shared with communication — see §2.4)

Each workspace session env preseeds a set of folders as candidate destinations for `move_files` / `archive_files`. Field contract is identical to `communication.folder`.

#### `type: "document"`

| Field | first-class | Notes |
|---|---|---|
| `title`, `doc_type`, `location`, `content_brief` | ✓ | `search_documents` filters by doc_type/location |

**Consumed by**: `search_documents`, `update_document`.

#### `type: "tracker"`

| Field | first-class | Notes |
|---|---|---|
| `title` | ✓ | |

**entries** can be preseeded under `attributes.entries` (a list of `{entry_id, status?, next_action?, payload?}`), or initialized empty and added dynamically by `update_tracker(update_action="create_entry", ...)`. **The tracker_id must already exist in references** — the agent cannot create a tracker from scratch. For `update_action="update_status"` / `close_entry` / `record_next_action` and similar operations that need an `entry_id`, the payoff should preseed entries under `attributes.entries` and tell the agent the entry_id via the instruction.

**Consumed by**: `update_tracker`.

---

## 3. General conventions

1. **id prefix**: follow §1's recommendation to differentiate by type prefix (`prod_* / rst_* / evt_* / ...`) — this is the data-construction convention from v0.10's `id_discovery_convention`, making it easier to spot a type misuse from an error message when the agent passes a flight id into a hotel slot.
2. **Don't put evaluative words inside `attributes`** (e.g. `"recommended": true` / `"best_choice": true`) — they leak the gold direction.
3. **`first-class` fields are the ones tools actually consume**; everything else lands in the `attributes` dict, and tools that the docstring explicitly notes as consumers (e.g. `price_changed` for commerce.subscription) still take effect.
4. **Empty `attributes` is legal**: `{"id":"x_1", "type":"product", "attributes": {}}` does not crash, but the product will use all-default values (price=0, category="unknown") — usually useless.
5. **Unknown `type` raises a runtime WARNING**: check the logs to find which reference was dropped.

---

## 4. Quick self-check

After authoring a batch of session_task JSON, load the env to verify all references are consumed:

```python
import logging, json
logging.basicConfig(level=logging.WARNING)
from runner.environment.base import PersonaProfile
from runner.environment.domains import build_env_for_session
from runner.schemas import SessionTask

task = SessionTask(**json.load(open("path/to/session_task.json")))
persona = PersonaProfile(persona_id="x")  # or load real persona
env = build_env_for_session(task, persona)

# Compare expected vs actual
expected = len(task.local_env.references)
actual = sum(len(getattr(env.toolkit.db, a)) for a in dir(env.toolkit.db)
             if isinstance(getattr(env.toolkit.db, a), dict) and a != "shopping_lists")
print(f"refs={expected}, lifted={actual}")
# If the counts don't match, some reference was dropped — scroll up through
# the WARNING logs to find which `type` was misspelled.
```
