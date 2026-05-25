# Rule QC

Score each rule on two independent axes (`counter_default` and `binding_sound`). The downstream keep / remove tag AND-s them — you only emit the per-axis labels and reasons. Score each rule independently; you do not need to consolidate duplicates, contradictions, or evidence quality across rules (all handled elsewhere). Score using `rule_text`, `counterfactual_default`, `check_type`, and `action_step`.

**Terminology.** *passive* = a generic helpful agent that does NOT know this rule. Note that `rule_text` is pipeline metadata; passive does NOT see it — at test time passive sees a neutral instruction generated from the rule + tool list + ontology defaults, so judge passive from the underlying scenario rather than from `rule_text` wording. *gold action* = the tool call this rule wants the agent to take (i.e. `action_step`).

---

## `counter_default` (yes | no)

> If we strip this rule away, would a generic helpful agent in the same scenario take the gold action anyway?

Yes → `counter_default = no` (rule is redundant, no personalized wedge). No → `counter_default = yes` (real wedge).

**NOT real wedges** (mark `no`). Current LLMs have a strong cautious / non-destructive default, so cautious-side rules are what passive does anyway: any "archive / pause / preserve / keep" rule that picks the safer option, and any "confirm before a destructive write" rule.

**OK wedges** (do NOT mark `no`). Rules that flip the passive default or pin a value passive would not reach for.

Example: "draft instead of send directly" — passive defaults to sending the prepared message; the rule flips that.

Example: "respond=tentative on conflict" — passive picks accept or decline; the rule pins the third enum value.

**Subtle failure.** Rule says "do X, don't do Y" but X and Y are not mutually exclusive — passive does both, so gold X is called even without the rule → `counter_default = no`.

Counter-example: "label community emails, don't archive" — labeling and archiving are independent steps; agents do both anyway.

### Examples for this axis

{{COUNTER_DEFAULT_POOL}}

---

## `binding_sound` (yes | no)

> Knowing this rule, would an agent naturally land on this exact `(check_type, action_step)`?

Yes → `binding_sound = yes`. No → `binding_sound = no`.

**Looks right but fails** (mark `no`):

- Rule is a cross-tool choice ("reminder rather than a new calendar event") but the binding lands on a param INSIDE one tool (e.g. `set_reminder.cadence`) — an agent following the rule would care about which tool, not which cadence.
- Rule talks about a dimension (time, formality, theme) but the binding is `tool_identity` with `param = null` — no parameter slot carries the dimension, so calling the tool does not express the preference.

**Looks wrong but actually OK** (do NOT mark `no`):

- Cross-tool wedge bound to the gold side — rule "prefer A over B", binding `A, tool_identity`. The rule's action IS calling A; the rejected B belongs in `counterfactual_default`, not in the binding.
- `param_id` rule whose dimension lives in candidate `attributes` — rule talks about time / theme / type, binding is `*_id`. The runtime populates each candidate's `attributes` with that dimension and the agent selects by reading them, so the binding carries the dimension indirectly.

**Enum catch-all warning.** Avoid `param_enum` whose enum includes an `"all"`-style catch-all. Agent safety bias favors the catch-all over the rule's specific value, making the rule unreliably testable.

### Examples for this axis

{{BINDING_SOUND_POOL}}

---

## Rules to review

```json
{{RULES_JSON}}
```

## Output format

```json
[
  {
    "rule_text": "<copy verbatim from input>",
    "counter_default": "yes" | "no",
    "counter_default_reason": "<one line; if 'no', name what passive would do by default; if 'yes', write 'passes'>",
    "binding_sound": "yes" | "no",
    "binding_sound_reason": "<one line; if 'no', name what the rule asks for vs what the binding lands on; if 'yes', write 'passes'>"
  }
]
```

Output count = input count. Every input rule appears exactly once.

Output JSON list only. No code fence. No other text.
