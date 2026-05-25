# Test Session Repair

The previous test session failed QC. Rewrite a new version (strict JSON, no code fence) such that, simultaneously:

- **Binary discrimination holds**: oracle hits gold; passive (commonsense) does NOT.
- The agent's first tool call hits the rule's relevant tool (the rule's decision point is not at a side step).
- Instruction is self-contained: every non-rule task parameter is written clearly.
- Refs hold only neutral facts (numbers / categories / source / identity / time / participants) — no value-judgments, recommendations, or actionability hints.
- For `confirm` rules, the instruction is an ordinary user request and must not leak the confirm flow (no "ask me first", "check with me", "confirm before", or two-step process narration).

Repair is NOT patching — you may change any field.

Cross-reference `failure_reasons` with the N `upper_reasons` / `lower_reasons` and their per-sample traces and match flags. Keywords repeated in passive self-reports point to where instruction or refs leak direction.

## Static-error precise fixes

- `instruction_leaks_ref_id:<ID>` → replace the literal id string with a natural-language description ("the train ticket from Florence to Venice" instead of `gt_florence_venice_20260512`).
- `action_{i}_id_missing_from_refs:<tool>.<arg>=<id>` → `gold_value` references an id not in `references`. Either add the object (`{id, type, attributes}`) or change `gold_value` to an existing id.
- `ref_type_invalid:<type>:domain=<domain>` → swap to a primary type from REF_SCHEMA's `#### type:` headers (NOT the `derived:` sections, which are auto-hydrated).
- `action_step_tool_is_exploration_prefix` → rule-level error; the repair stage cannot fix it — return to rule generation.

## Reading the QC failure mode → targeted fix

- **Mode A** (`upper_matched = False`): oracle did not hit gold. Causes: refs lack an object satisfying the rule direction (fix `references` so exactly 1 ref satisfies every dimension); `gold_value` is not what oracle would pick (fix `gold_value`); or the instruction's task parameters conflict with the gold ref's attributes ("Beijing restaurant" but gold ref is Shanghai — align one with the other). Read `upper_reasons` for the sentence where oracle says why it picked X instead of gold.

- **Mode B** (`upper_matched = True, lower_matched = True`): the main failure mode — passive also hit gold. Find repeated keywords in `lower_reasons`:

  - "I picked X because the instruction said Y" → instruction leak. Apply the wording-neutrality guidance from the generation prompt (drop the type / action / container / cadence / business-type / replacement-need / loaded-scenario leak).
  - "I picked X because its attribute Z fits best" → refs let gold's rule-direction attribute be too prominent. Apply the refs-as-a-whole guidance from the generation prompt: give decoys pull attributes (price / rating / popularity / official / newest / closest / most full-featured); for `param_id`, if gold is exclusive on the rule attribute, the decoys must beat gold on a secondary dimension; for `tool_identity`, refs must include candidates the counterfactual tool would handle.

- **Mode C** (`upper_matched = False, lower_matched = True`): both wrong. Usually gold or markers were set wrong — fix oracle per Mode A; if the lower problem does not disappear as a side effect, apply Mode B.

## Real-failure reminders — do not repeat

- `param_id` · gold-exclusive on the rule direction → add 1–2 same-direction decoys that beat gold on a secondary dimension.
- `param_id` · numeric gap too obvious (gold rating 4.9 vs. decoys topping out at 4.7): LLMs lock onto "highest rating" = gold; gold rating must NOT be the maximum.
- `param_enum` · task-word leak ("project" ≈ `by_project`; "key" ≈ critical): neutralize phrasing.
- Rule direction == LLM default: fundamentally unconstructible — do not force-fix, report "unconstructible".

## Few-shot examples (same / similar bucket)

{{FEW_SHOT_EXAMPLES}}

---

## Inputs

### Previous test session

```json
{{CURRENT_TASK}}
```

### Failure reasons

```json
{{FAILURE_REASONS}}
```

### Rule

```json
{{RULE_JSON}}
```

### Domain = `{{DOMAIN}}` available tools

```json
{{DOMAIN_TOOL_SPECS}}
```

### References format contract

{{REF_SCHEMA}}

---

## Hard MUSTs (violation → entire repair is discarded)

Retain these 3 fields; values may change but keys cannot be dropped:

1. `instruction` (str, non-empty).
2. `references` (≥3 objects for `param_id`; smaller is OK for other check_types if there is at least 1 gold-context object).
3. `gold_value` (per `rule.check_type`):
   - `tool_identity` (mutate, `action_step.param = null`) → `null`
   - `param_id` → `"<gold ref id>"` (must exist in `references`)
   - `param_enum` → `"<enum value>"` from the enum's value set
   - `confirm` → `null` (confirm-only) or `{"<inner_param>": <value>}` (inner_param must be safe-typed on `rule.action_step.param`'s tool signature)

Do NOT output `domain`, `local_env.tools`, `labels`, or `session_id` — the caller derives them. A common past mistake was emptying or mistyping `gold_value`; do not repeat it.

Output the complete repaired test session JSON now, no markdown code fence.
