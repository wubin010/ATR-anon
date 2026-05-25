# Rule Generation

From the persona below, produce {{N_RULES}} standing rules. Each rule is one sentence describing a long-term preference of this user and must map cleanly to one tool call from the tool list.

## What makes a good rule

1. **Grounded in the persona.** Point to a specific phrase, behavior, activity, occupation, or community. Generic adjectives ("creative", "diligent", "careful") do not count — they fit anyone.

2. **Surprising default.** A good rule flips what a generic helpful agent would otherwise do; if the agent would take the same action anyway, the rule is dead weight.

   Today's LLM agents have a strong cautious / non-destructive bias, so cautious-side wedges (archive-over-delete, pause-over-cancel, confirm-before-destructive-write, prioritize-important, track-after-ordering) are NOT surprising — they are the default.

   Example: "Delete promo emails outright." (Agents default to archive; this rule pushes the destructive action.)

3. **Maps to one tool call.** See `check_type` below.

Skip candidates that do not cleanly map (feelings, internal moods, things the tool list cannot express).

## Output

JSON list. Each entry:

```json
{
  "rule_text": "<third-person sentence; do not name tools or parameters>",
  "canonical_answer": "<first-person sentence that stands on its own (e.g., 'I always X for Y' / 'When Y, I do Z' / 'For X, I prefer Y'); do not name tools or parameters>",
  "counterfactual_default": "<what someone without this rule would naturally do>",
  "evidence": [
    {"type": "direct",   "content": "<exact phrase from the persona>"}
    // OR {"type": "inferred", "content": "<reasoning chain rooted in specific persona facts>"}
  ],
  "check_type": "tool_identity | param_id | param_enum | confirm",
  "action_step": {
    "tool":  "<exact tool name from tool list>",
    "param": "<param name on that tool, OR null, OR — for confirm rules — the inner mutate tool name>"
  }
}
```

## `check_type` — pick one of four

- **`tool_identity`** — pins WHICH TOOL. `param = null`.
- **`param_enum`** — pins an ENUM VALUE on an enum-typed param.
- **`param_id`** — pins WHICH ITEM (by attribute) on a `*_id` / `*_ids` parameter. Not for ids the user hands in ("delete file X" — that is instruction following).
- **`confirm`** — asks for `get_user_confirmation` before a mutate; `param` carries the INNER mutate tool name (NOT a parameter on `get_user_confirmation`). A confirm rule may also pin an inner-param preference; that surfaces later as test-session `gold_value`, not in `action_step`.

## Hard rules

- Tool and param names are exact strings from the tool list.
- `param` is what the rule pins (output choice), not what the user hands in (input id). For instance, pin `selected_alternative_stop_ids`, not `disrupted_item_id`.
- Do not put a *rejected* tool in `action_step`. For "draft instead of send", `tool=draft_message`, never `send_message` — the rejected tool belongs in `counterfactual_default`.
- For "do X, don't do Y" rules, X and Y must be alternatives — calling X precludes Y, or Y is something nobody would call here.

  Example: "archive instead of delete" — archive and delete are exclusive cleanup actions.

  Counter-example: "label community emails, don't archive" — labeling and archiving are independent steps, so the agent does both anyway.

- No duplicate or contradicting bindings. "Direction" = which side the rule prefers. At most one rule per `(tool, param, direction)`. Two rules on the same `(tool, param)` are OK only if their directions are orthogonal axes (one pins cuisine, another pins seating — both can hold on one restaurant call). Two rules pinning the same trigger to opposite values: drop the weaker one.

### Tool pairs that look similar — pick deliberately

- `plan_trip` (new) vs `replan_trip` (existing trip; needs `trip_id` + `disrupted_item_id`).
- `modify_event` (anything except time) vs `reschedule_event` (time only).
- `update_tracker` (append to existing; needs `tracker_id`) vs `create_document` (new doc / spreadsheet / note).

## Examples

```json
// tool_identity
{"rule_text": "When cleaning up work files, archive rather than delete.",
 "check_type": "tool_identity",
 "action_step": {"tool": "archive_files", "param": null}}

// param_enum
{"rule_text": "When an invite conflicts with something unresolved, respond tentatively rather than declining.",
 "check_type": "param_enum",
 "action_step": {"tool": "respond_to_event_invite", "param": "response"}}

// param_id
{"rule_text": "For garden purchases, favor locally-adapted native plants over generic landscaping options.",
 "check_type": "param_id",
 "action_step": {"tool": "place_order", "param": "product_id"}}

// confirm
{"rule_text": "When new fad-style health services come up, check with her before booking.",
 "check_type": "confirm",
 "action_step": {"tool": "get_user_confirmation", "param": "book_service_appointment"}}
```

## Tool list

{{TOOLS_TEXT}}

## Persona

{{STRUCTURED_PERSONA}}

## Final per-rule checks

1. `counterfactual_default` names a *different action*, not just less of the same.
2. The same persona trait is not already covered by another rule in your output.

Output JSON list only. No code fence. No other text.
