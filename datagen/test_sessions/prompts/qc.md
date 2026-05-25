# Test Session QC — Agent Trace Simulation

You are an assistant. Complete the user's task by calling the available tools. Output the full trace in a single response — list every tool call you intend to make, in order. Do not split across turns; do not wait for tool returns.

{{RULE_CONTEXT_BLOCK}}

## Task

### User request

{{INSTRUCTION}}

### Available tools

```json
{{TOOL_SPECS}}
```

### Objects in the environment

```
{{REFERENCES}}
```

> Candidate objects you may reference by id. Pick directly — do NOT add a `search_*` / `list_*` discovery prologue.

## Conventions

- List tool calls in natural execution order.
- Argument values must conform to the schema: `enum` values come from the listed set (never invent); `boolean` is `true` / `false`; other `string` params must not be fabricated; `id` params may reference only the ids listed above.
- Always supply required parameters; for optional parameters, supply a value only when the user request, the environment objects, or an applicable user preference clearly determines it.

## Output format (strict JSON, no code fence)

```jsonc
{
  "trace": [{"tool": "<tool_name>", "arguments": {<kwargs>}}, ...],
  "reason": "one sentence explanation"
}
```

Output now.
