# Test Session Generation

Generate one test session for the given rule. At runtime the session is user-offline — the agent executes alone, with no one to clarify with — so the instruction must be self-contained and must NOT reveal the rule direction.

The session is valid iff the binary discrimination holds:

- **oracle** (knows the rule) hits gold tool + gold args.
- **passive** (does not know the rule), reading the same instruction + refs and reasoning by commonsense, does NOT hit gold.

Use `rule.counterfactual_default` as your prediction of passive's behavior and point your decoy design that way so the oracle / passive gap stays sharp.

This is a single-call task — no refine loop. Reason through the steps below internally, then emit the final JSON.

---

## Locate the discriminative dimension

From `rule_text` + `canonical_answer` + `counterfactual_default`: what does oracle do (gold)? what does passive do (counterfactual)? where is the difference — `param_id` (which ref selected) / `param_enum` (which enum value) / `tool_identity` (which tool)?

## Refs design

### Per-decoy pull (param_id only)

Plan 1 gold + ≥2 decoys. Every decoy needs at least one concrete *pull attribute* — a property that, on its own, attracts a rule-blind agent. Menu: lowest price / highest rating / most popular / official or authoritative / most recently updated / closest distance / most full-featured.

- Decoy check: "Without the rule, which attribute pulls toward this decoy?" If you cannot name it, redesign — a parallel choice with no pull leaks because passive picks at random.
- Gold check: "With the rule, does the rule-aware agent uniquely lock on gold?" If a decoy also satisfies the rule + instruction demands, oracle could legitimately pick it — redesign.

For `param_enum` / `tool_identity`, refs still need ≥1 gold + 1–2 decoys for context, but gold determination depends on the enum value / tool, not on the ref set.

### Refs-as-a-whole must lean toward the counterfactual

Read `counterfactual_default`, work backwards to what signal in refs points that way, then amplify it. The rule-direction signal should be the *least* prominent.

- **tool_identity** (rule = `create_document`, counterfactual = `update_tracker`): refs must include 1–2 appendable-looking trackers so passive leans toward update.
- **param_enum** (rule = `scheme=by_project`): each ref carries the rule dimension AND other dimensions (`created_date`, `doc_type`) so passive has multiple classification paths.
- **param_id**: the rule-direction attribute must NOT be gold-exclusive. Give ≥2 candidates with the rule's direction (1 gold + ≥1 decoy), and let decoys beat gold on a secondary dimension (rating, distance, popularity) so passive picking by secondary signal lands on a decoy.

  Counter-example. Rule "same theme (nature)" + refs = 1 nature garden + 3 art / spa / cafe. Passive sorting "same theme" has no choice but gold.

  Example. Gold = garden (nature); decoy1 = overlook (nature, far); decoy2 = temple (cultural, 4.8 rating); decoy3 = cafe (4.9). Two nature candidates with secondary-dimension reversal.

Self-check: count refs carrying each rule-direction attribute. Only 1 (gold-exclusive) → add a same-direction decoy.

### Two more ref traps

- **Hard-constraint exclusivity** (`param_id`). If instruction lists hard constraints ("Old Harbour + 2 adults + private bathroom"), every decoy must satisfy all of them, else instruction-filtering rejects them and gold becomes the only valid pick. Differentiate decoys on secondary dimensions, not on the hard constraints.
- **Project-lifecycle naming** (`param_enum scheme=by_project`). Ref names that read like one project's stages (survey → costs → recap → packet) commonsense-pin passive on `by_project` even with neutral instruction. Mix names across multiple visible projects, vary `file_type`, and spread `created_date` over months.

## Gold triple-filter

All three must be yes before fixing gold:

1. **Rule-conformant**: gold satisfies every dimension of the rule direction. (Do not misread "A over B" — gold must BE A, not just "not B".)
2. **Instruction-conformant**: gold satisfies the instruction's explicit non-rule demands (time, location, count, budget).
3. **Decoy disadvantage**: every decoy fails at least one rule or instruction dimension strictly worse than gold; no decoy simultaneously satisfies 1 + 2.

## Instruction

Write a natural one-paragraph user request. Include the non-rule task parameters; do NOT include rule-direction words, the gold ref id, or the gold tool name.

### Wording neutrality (the main passive-leak source)

For every concrete noun and verb, ask: "On its own, does this word push passive toward gold?" If yes, swap for neutral phrasing. Common leak patterns and the neutral rewrites:

- **Type-word** ("project files" when rule is `by_project`) — neutralize to "random files that have piled up".
- **Action-word** ("draft a message" when rule is draft-over-send) — neutralize to "send X a quick note".
- **Container / location-word** ("don't leave them in the workspace" when rule is archive-over-delete) — neutralize to "clean them up".
- **Frequency / cadence-word** ("every week I do this" when rule is weekly cadence) — neutralize to one absolute time ("Wednesday at 8pm"), and let oracle infer weekly from the rule.
- **Business-type word** ("claims, renewals, policy-change files" when rule is `by_project`) — neutralize to "files I've collected over the past few months".
- **Replacement-need word** ("the compact one won't fit" when rule is exchange-over-refund and refs hold obvious replacements) — neutralize to "I want to send this back".
- **Loaded-scenario** (long-haul "Seattle → London, pick a seat" pre-commits passive to `window`; broken-item scenarios pre-commit `resolution_type='exchange'`; deadline-loaded messages pre-commit `priority='high'`; one-off scheduled tasks pre-commit `cadence='before_event'`) — replace with a neutral scenario, or omit the cue field entirely.

Rule of thumb: as passive, read instruction alone — does any word make you think "ah, must be option X"? If X = gold, drop the word. Legitimate task parameters (time, location, headcount, target person) must still be written clearly — these are task context, not direction words. Test: remove the word; does the task still make sense? Yes → keep; No → it was a direction word, drop.

### Discoverability

For every ref id in `gold_value` (or in `*_id` / `*_ids` slots inside `gold_value` for confirm rules), the instruction must contain at least one token that loose-matches a token in one of the ref's searchable attributes (`name`, `title`, `subject`, `sender`, `location`, `region`, `city`, `address`, `destination`, `origin`, `cuisine`, `service_type`, `category`, `doc_type`, `tags`). Without this, the agent's search returns 0 results and the session is unsolvable.

Loose-match = after lowercasing and splitting on whitespace / punctuation, some instruction token equals or is a substring of some attribute token. If persona / refs use English place names while the instruction is Chinese (or vice versa), plant the English token verbatim somewhere in the instruction.

### Time handling

ATRBench does not model a time axis: search tools do NOT take time / date filters. The agent fetches refs first, then reads `date_time` / `start_time` / `departure_date` on each candidate.

- Default: no specific time in instruction ("book dinner for me").
- Relative time is forbidden: never write "next Thursday / this week" or any non-Latin-script equivalent — runtime hides the current date.
- Absolute date is allowed only for time-related rules (cadence enum, time-triggered notification, scheduling-conflict pattern): use `YYYY-MM-DD`, and the gold ref's date field must contain this date as a substring (`"2026-05-18T15:00:00"` matches `2026-05-18`).

When in doubt, omit time.

### Derived entities

When the gold tool operates on a derived entity (orders / trips / reservations / bookings / appointments), the env hydrates the entity from the corresponding primary ref's attributes (see `derived:` sections in REF_SCHEMA for which primary ref carries which ID attribute). Place the derived ID on the right primary ref's `attributes`, else gold fails with "not found".

In the instruction, describe the entity by role / context, never by its specific name, and never with a literal id (`ord_xxx` / `rst_xxx`).

Example. "Return the candle I ordered last month." (Role.)

Counter-example. "Return the Cedar Glow Candle I ordered." (Specific name — passive can match it directly.)

---

## Few-shot examples (same / similar bucket)

{{FEW_SHOT_EXAMPLES}}

---

## Counter-example — real failure, do not repeat

`param_id` · numeric pull too obvious. Rule "prefer high-rating reliability over distance". Wrong refs: gold = Cascade Code (rating = 4.9, dist = 6.4 km); decoys top out at 4.7 rating with distances under 1 km. Passive locks onto "highest rating" = gold because LLMs are far more sensitive to rating jumps (4.7 → 4.9) than to distance gaps (0.5 → 6.4 km). Fix: mix gold's rating into the middle pack, and let a decoy own "highest rating + closest" so passive is pulled away.

---

## Output JSON schema

You write 3 fields:

```json
{
  "instruction": "<one English paragraph; non-rule task params present, no rule-direction words>",
  "gold_value": <see dispatch table below>,
  "references": [{"id": "...", "type": "...", "attributes": {...}}]
}
```

You do NOT write `domain`, `local_env.tools`, `session_id` / `session_type` / `persona_id` / `rule_id` / `rule_ref`, or `labels` — the caller derives them.

### `gold_value` dispatch (by `rule.check_type` × param type)

Find `<rule.param>`'s type in the tool signature inside `DOMAIN_TOOLS`.

| `check_type` | `<rule.param>` type | `gold_value` |
|---|---|---|
| `tool_identity` | (n/a, no param) | `null` |
| `param_id` | `string` (single id) | `"<ref_id>"` |
| `param_id` | `array[string]` (list of ids) | `["<ref_id>", ...]` (typically length 1) |
| `param_enum` | `enum[...]` | `"<enum_value>"` |
| `confirm` | (param = inner mutate tool name) | `null` (confirm-only) OR `{<inner_param>: <inner_value>}` |

For `array[string]` params, list the gold ids (typically one, since refs
hold exactly one satisfying object); the evaluator matches them by set
equality.

### Confirm rule details

`rule.action_step.tool` is always `get_user_confirmation`; `rule.action_step.param` carries the inner mutate tool name (e.g. `"delete_files"`, `"book_service_appointment"`).

- No inner-param preference (purely "confirm vs not"): `gold_value = null`. Caller fills `arguments = {target_tool: <inner>}` and `compare_args = ["target_tool"]`.
- Inner-param preference (e.g. "ask before canceling the recently-added appointment"): `gold_value = {<inner_param>: <value>}`. Caller adds `target_params: <gold_value>` and `compare_args = ["target_tool", "target_params"]`. The inner param must exist on the inner mutate tool's signature with a safe type (`*_id` / `*_ids` / `enum` / `boolean`).

---

## Hard self-check (run before emitting)

1. Non-rule task params (location, headcount, target person) are written clearly in the instruction.
2. References contain exactly 1 object satisfying the rule direction for `param_id` (and inner-`param_id` for confirm rules).
3. `gold_value` matches `rule.check_type` × `<rule.param>` type per the dispatch table.
4. Ref attributes hold only neutral facts (numbers, categories, identity, time, participants) — no value-judgments, recommendations, or actionability hints.

### Confirm rule special constraint

If `rule.check_type == "confirm"`, the instruction is an ordinary user request ("help me schedule X" / "handle it") and must NOT contain any wording about a confirm flow ("ask me first", "check with me", "confirm before", "get my permission", "decide later") or any two-step process narration ("first X, then Y"). The confirm step is what oracle infers from the rule — if instruction leaks it, passive also calls `get_user_confirmation` and the discrimination disappears.

### id-reference hard red lines (static check enforces)

1. Any `*_id` / `*_ids` value in `gold_value` must be the id of an object in `references`. No fabricated ids.
2. `references[i].type` must be a primary type listed in REF_SCHEMA (the `type:` headers — not the `derived:` sections).

---

## Inputs

### Rule

```json
{{RULE_JSON}}
```

### Domain = `{{DOMAIN}}` available tools

```json
{{DOMAIN_TOOLS}}
```

### Domain `{{DOMAIN}}` reference schema

{{REF_SCHEMA}}

---

Emit strict JSON now, no markdown code fence.
