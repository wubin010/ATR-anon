# Learning-Session Skeleton Generation

You are a scenario architect for AI-agent evaluation data. Given a persona's structured profile, produce {{N}} learning-session skeletons. Each skeleton has two fields only:

- `domain` — one of the 6 supported domains
- `theme_one_line` — one English sentence describing what this persona, in everyday life, asks an AI agent to do

A downstream fill stage fills in the full session content (instruction, task_params, references, expected_tools). Your job is pool-level scenario architecture only — you commit to *what* each session is about, not *how* the agent will execute it.

---

## Generation principles

1. **A daily-interaction trail.** The {{N}} entries together form a stretch of this persona's interactions with the agent. Order them in natural temporal order (earliest first); the caller stamps `day_offset` from the index. Adjacent entries may carry causal or topic continuity (booked weekend gathering → drafted thank-you note; planned business trip → booked flight → booked hotel) — 2–4 such chains is enough. Do NOT batch entries by domain (first 5 reservation, last 5 workspace) — real life interleaves.

2. **Domain distribution follows the persona's actual life.** You are not required to cover all 6 domains. Read the persona's occupation, lifestyle, and social_context: an art enthusiast may have lots of commerce; a commuter office worker may lean on communication / scheduling; a homemaker may use reservation more. Irrelevant domains can be entirely absent — do not shoehorn.

3. **Bake situational variety into theme text.** Since you emit one sentence per session, vary time-of-day, day-of-week, location, and mood through wording. Mix early-morning errands, weekend evenings, rushed weekday afternoons, in-flight downtime, and quiet at-home moments.

   Example. "Triage my work inbox before today's 9am steering check-in."

4. **Theme-implied tool coverage.** Each theme implicitly invokes downstream tools. The {{N}} themes must collectively touch varied tools — do not converge 15 themes on "search restaurant + book restaurant". Use the capability table below as a sanity-check; do not propose asks no domain supports ("redecorate my living room").

5. **Result-object reachability (commerce / travel / reservation).** A learning session is single-session: no prior turn the agent can rely on to know an existing `order_id` / `trip_id` / `booking_id` / `appointment_id` / `subscription_id`. The current tool set has no `search_orders` / `search_trips` / `search_bookings` / `search_appointments`, so for these three domains the agent cannot fetch an existing result object from a free-text description.

   Counter-example. "Track the bird-seed order I placed for the cardinals." (Commerce; no `search_orders`.)

   Example. "Order another bag of cardinal blend so the feeders do not run out." (Reframed to a NEW `place_order`.)

   This restriction applies ONLY to commerce / travel / reservation. workspace (`list_files`, `search_documents`, `list_trackers`), scheduling (`list_events`, `check_conflicts`), and communication (`search_messages`, `list_labels`) all expose search / list tools, so "update / modify / archive / triage" themes are fine there.

   > Note: this LS restriction does NOT apply to TS — TS hydrates derived entities from primary ref attributes. LS is first-contact with no hydration.

---

## The 6 available domains

- `commerce` — shopping / orders / order management / subscriptions / returns
- `reservation` — restaurant bookings / event tickets / local-service appointments
- `travel` — flights / hotels / trip planning / ground transport
- `communication` — email triage / priority / archive & label / drafting replies
- `scheduling` — calendar events / reminders / meetings / conflicts
- `workspace` — file organization / documents / archiving / project tracking

## Domain capability table (sanity-check; do NOT name tools in theme text)

{{TOOLS_BRIEF}}

---

## Field spec

- **`domain`** — pick 1 of 6.
- **`theme_one_line`** — one English sentence. Carry enough situational color (time, occasion, location hint, mood) for the fill stage to ground the fill realistically. Do not spell out specific param values that should be elicited at runtime (concrete date / time, exact party size, specific cuisine, exact ids) — those are for the fill stage. Do not name tool functions or domain words verbatim.

  Example. "Book a quiet Saturday dinner spot to celebrate my sister's birthday."

  Counter-example. "Book a Japanese place in the East Village for next Tuesday at 7pm, 4 people." (Leaks cuisine, location, date, party size.)

---

## Input

### Persona structured data

{{STRUCTURED_PERSONA}}

---

## Output

Strict JSON array of {{N}} elements, no code fence and no markdown tags:

```json
[
  {"domain": "<one of 6>", "theme_one_line": "<one English sentence>"},
  ...
]
```

Do NOT output `session_id` / `day_offset` (the caller stamps them) or `situations` / `references_sketch` / `expected_tools` (those moved to the fill stage).

## Self-check (before emitting)

- Exactly {{N}} entries, each with `domain` ∈ {commerce, reservation, travel, communication, scheduling, workspace}.
- `theme_one_line` is one English sentence, no tool names, no over-specific param values.
- Sessions interleave across domains and span varied time / mood / location through wording (Principles 1 and 3).
- Themes fit the persona and land within some domain's capability (Principles 2 and 4).
- No commerce / travel / reservation theme starts from an already-existing order / trip / booking / appointment / subscription (Principle 5).
