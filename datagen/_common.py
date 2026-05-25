"""Shared utilities for datagen stages (ingest / rule generation / rule QC /
learning-session skeleton / learning-session fill / test-session / episode compose).

Provides ontology loaders, tool-schema rendering, ref-schema
extraction, prompt IO, small JSON helpers. Persona-specific helpers
live next to the stage that owns them (ingest for persona_process,
session_gen/fill for structured-persona-to-brief rendering, etc.).
"""
from __future__ import annotations

import json
import logging
import re
import sys
import typing
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONTOLOGY_PATH = PROJECT_ROOT / "ontology" / "tools.yaml"
REF_SCHEMA_PATH = PROJECT_ROOT / "docs" / "REF_TYPE_SCHEMA.md"
PERSONAS_DIR = PROJECT_ROOT / "data" / "personas"

# Side-effect: make `from lib.llm import ...` and `from runner.... import ...`
# work from every stage without each one repeating the sys.path dance.
sp = str(PROJECT_ROOT)
if sp not in sys.path:
    sys.path.insert(0, sp)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ontology loaders (cached)
# ---------------------------------------------------------------------------

_ONTOLOGY_CACHE: dict | None = None
_TOOL_MAP_CACHE: dict[str, dict] | None = None
_DOMAIN_TOOLS_CACHE: dict[str, list[dict]] | None = None

# ATR v2 domains — mirrors runner/environment/domains/*
DOMAINS = ("commerce", "reservation", "travel", "communication", "scheduling", "workspace")


def load_ontology() -> dict:
    global _ONTOLOGY_CACHE
    if _ONTOLOGY_CACHE is None:
        with open(ONTOLOGY_PATH, encoding="utf-8") as f:
            _ONTOLOGY_CACHE = yaml.safe_load(f)
    return _ONTOLOGY_CACHE


def tool_map() -> dict[str, dict]:
    """Name → tool dict (with 'domain' key added).

    Base tools (only `get_user_confirmation` in MVP) from the ontology's
    `base_tools:` section are registered with domain="base" so cross-domain
    validators treat them as domain-agnostic.
    """
    global _TOOL_MAP_CACHE, _DOMAIN_TOOLS_CACHE
    if _TOOL_MAP_CACHE is not None:
        return _TOOL_MAP_CACHE
    local_map: dict[str, dict] = {}
    local_domains: dict[str, list[dict]] = {}
    data = load_ontology()
    for domain_name, domain in data["domains"].items():
        local_domains[domain_name] = []
        for tool in domain["tools"]:
            item = dict(tool)
            item["domain"] = domain_name
            local_map[tool["name"]] = item
            local_domains[domain_name].append(item)
    for tool in (data.get("base_tools", {}) or {}).get("tools", []) or []:
        item = dict(tool)
        item["domain"] = "base"
        local_map[tool["name"]] = item
    _DOMAIN_TOOLS_CACHE = local_domains
    _TOOL_MAP_CACHE = local_map
    return _TOOL_MAP_CACHE


def base_tool_names() -> set[str]:
    return {name for name, t in tool_map().items() if t.get("domain") == "base"}


def domain_tools(domain: str) -> list[dict]:
    tool_map()  # warm cache
    return list(_DOMAIN_TOOLS_CACHE.get(domain, []))  # type: ignore


def domain_of_rule(rule: dict) -> str | None:
    """Derive domain from the rule's gold tool.

    Confirm rule (action_step.tool == get_user_confirmation): use
    action_step.param (the target mutate tool name) to look up the
    business domain. Other rules: action_step.tool's own domain.
    """
    tmap = tool_map()
    step = rule.get("action_step") or {}
    if not isinstance(step, dict):
        return None
    if rule_is_permission(rule):
        target = step.get("param")
        if target:
            d = tmap.get(target, {}).get("domain")
            if d and d != "base":
                return d
        return None
    tname = step.get("tool")
    if not tname:
        return None
    return tmap.get(tname, {}).get("domain") or None


def rule_is_permission(rule: dict) -> bool:
    """Confirm-style (permission) rule.

    First-class signal: `rule.check_type == "confirm"`. Falls back to
    inspecting `action_step.tool == "get_user_confirmation"` for data
    without a `confirm` check_type (validator enforces both in lockstep).

    Test sessions are user-offline; the agent's duty ends at issuing the
    confirm call (no user reply arrives). Downstream:
      - `derive_required_actions` produces a single confirm RequiredAction
        whose target_tool comes from action_step.param and target_params
        from gold_value (when present)
      - gen/refine prompts warn against leaking the confirm into
        `instruction`
    """
    if rule.get("check_type") == "confirm":
        return True
    step = rule.get("action_step") or {}
    if not isinstance(step, dict):
        return False
    return step.get("tool") == "get_user_confirmation"


# ---------------------------------------------------------------------------
# Tool schema rendering (for prompts)
# ---------------------------------------------------------------------------


def _enum_choices(type_name: str) -> list[str]:
    m = re.fullmatch(r"enum\[(.+)\]", type_name)
    return [s.strip() for s in m.group(1).split("|")] if m else []


def _tool_to_openai_schema(
    tool: dict,
) -> dict:
    """Render one tool to OpenAI function-calling JSON schema (prompt-friendly).
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in tool["parameters"]:
        t = str(p["type"])
        name = p["name"]
        entry: dict[str, Any] = {}
        if t == "string":
            entry = {"type": "string"}
        elif t == "integer":
            entry = {"type": "integer"}
        elif t == "number":
            entry = {"type": "number"}
        elif t == "boolean":
            entry = {"type": "boolean"}
        elif t == "date":
            entry = {"type": "string", "format": "date"}
        elif t == "datetime":
            entry = {"type": "string", "format": "date-time"}
        elif t == "array[string]":
            item_entry: dict[str, Any] = {"type": "string"}
            entry = {"type": "array", "items": item_entry}
        elif t.startswith("enum["):
            entry = {"type": "string", "enum": _enum_choices(t)}
        else:
            entry = {"type": "string"}
        props[name] = entry
        if p.get("required"):
            required.append(name)

    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": {"type": "object", "properties": props, "required": required},
    }


def render_domain_tools(domain: str) -> str:
    """Render a single domain's tools as a JSON-schema array (prompt block)."""
    specs = [_tool_to_openai_schema(t) for t in domain_tools(domain)]
    return json.dumps(specs, ensure_ascii=False, indent=2)


def render_all_domain_tools_brief() -> str:
    """Render all domains' tools as a compact list (used by stage C / B
    prompts that need the full tool surface, including read-prefix
    exploration tools the agent will actually call at runtime).

      ### DOMAIN
        - tool_name (param:type*, ...) — short description
    """
    data = load_ontology()
    lines: list[str] = []
    for domain_name, domain in data["domains"].items():
        lines.append(f"### {domain_name.upper()}")
        for tool in domain["tools"]:
            params = ", ".join(
                f"{p['name']}:{p['type']}" + ("*" if p.get("required") else "")
                for p in tool["parameters"]
            )
            lines.append(f"  - {tool['name']} ({params}) — {tool['description']}")
        lines.append("")
    return "\n".join(lines)


# Read-prefix tools cannot be the gold tool of a rule. Two reasons:
#   (a) hard reads — passive agents call them naturally during commonsense
#       exploration and trivially "hit gold" (search/list/lookup/get/...)
#   (b) monitoring / checking — even when passive doesn't always call them,
#       any rule pinning their enum params (e.g. update_scope) almost
#       always has the catch-all "all" option, triggering the
#       framework_blind_spot:A trap (oracle's safety bias picks "all"
#       instead of the rule's specific value)
# `get_user_confirmation` is the one base-domain tool that IS rule-bindable
# (confirm rules), surfaced in a dedicated section below.
_RULE_NON_BINDABLE_PREFIXES = (
    # hard reads — passive must call
    "search_", "list_", "lookup_", "get_", "view_", "browse_", "compare_",
    "shortlist_", "filter_", "rank_", "refine_", "narrow_",
    # checks / finds — passive calls these by commonsense too
    "check_", "find_",
    # monitoring / reviewing — banned to avoid framework_blind_spot:A on
    # their enum params (update_scope etc. all carry catch-all values)
    "track_", "review_",
)


def runtime_entity_tools() -> set[str]:
    """Tools whose target entity lives in a runtime-only TableSpec
    (created by write tools, never hydrated from refs).

    Rules bound to such tools produce unexecutable test sessions: TS is
    user-offline single-step, so without a discoverable ref the entity
    can't exist at test time. Both `render_bindable_tools` (hide from
    LLM) and `rules.gen` validator (reject if LLM still proposes one)
    consume this set — single source-of-truth lookup.
    """
    from runner.environment.tables import all_table_specs
    out: set[str] = set()
    for _, spec in all_table_specs():
        if spec.kind == "runtime" and spec.operating_tools:
            out.update(spec.operating_tools)
    return out


def render_bindable_tools() -> str:
    """Render tools that can be the GOLD tool of a rule.

    Scope:
      - mutate-style tools across all business domains (anything that
        doesn't start with a read prefix)
      - `get_user_confirmation` (base domain, special — confirm rules)

    Filtered out:
      - read-prefix tools (`_RULE_NON_BINDABLE_PREFIXES` above)
      - runtime-only tools (`runtime_entity_tools()` — TS unexecutable)
      - boolean-typed parameters within otherwise-bindable tools
        (plan_trip's include_* flags etc. — output-format toggles, not
        actionable choices)
    """
    data = load_ontology()
    runtime_tools = runtime_entity_tools()
    lines: list[str] = []
    for domain_name, domain in data["domains"].items():
        bindable = [
            t for t in domain["tools"]
            if not t["name"].startswith(_RULE_NON_BINDABLE_PREFIXES)
            and t["name"] not in runtime_tools
        ]
        if not bindable:
            continue
        lines.append(f"### {domain_name.upper()}")
        for tool in bindable:
            params = ", ".join(
                f"{p['name']}:{p['type']}" + ("*" if p.get("required") else "")
                for p in tool["parameters"]
                if p.get("type") != "boolean"
            )
            lines.append(f"  - {tool['name']} ({params}) — {tool['description']}")
        lines.append("")

    # Surface get_user_confirmation explicitly (base domain otherwise hidden).
    confirm = tool_map().get("get_user_confirmation")
    if confirm:
        params = ", ".join(
            f"{p['name']}:{p['type']}" + ("*" if p.get("required") else "")
            for p in confirm["parameters"]
            if p.get("type") != "boolean"
        )
        lines.append("### CONFIRM (cross-domain — for confirm rules)")
        lines.append(
            f"  - get_user_confirmation ({params}) — {confirm['description']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ref schema rendering (TableSpec source of truth)
# ---------------------------------------------------------------------------


def ref_types_for_domain(domain: str) -> set[str]:
    """Legal ref_type strings for a domain (from TableSpec source of truth)."""
    from runner.environment.tables import specs_for_domain
    out: set[str] = set()
    for spec in specs_for_domain(domain):
        if spec.kind == "primary" and spec.source_ref_type:
            out.add(spec.source_ref_type)
            out.update(spec.aliases or [])
    return out


def render_ref_schema(domain: str) -> str:
    """Render domain ref schema from TableSpec source of truth.

    Replaces the old REF_TYPE_SCHEMA.md section extraction with dynamic
    rendering from runner.environment.tables declarations.
    """
    from runner.environment.tables import specs_for_domain
    specs = specs_for_domain(domain)
    if not specs:
        return f"(no ref types registered for domain '{domain}')"

    lines: list[str] = []

    for spec in specs:
        if spec.kind == "primary":
            _render_primary_ref_type(spec, lines)
        elif spec.kind == "derived":
            _render_derived_ref_type(spec, lines)

    return "\n".join(lines)


def _render_primary_ref_type(spec, lines: list[str]) -> None:
    ref_type = spec.source_ref_type
    header = f'#### `type: "{ref_type}"`'
    if spec.aliases:
        alias_str = " or ".join(f'`type: "{a}"`' for a in spec.aliases)
        header += f" (or {alias_str})"
    lines.append(header)
    lines.append("")

    if spec.model and spec.promoted_attrs:
        fields = spec.model.model_fields
        lines.append(
            "Every field except `id` / `type` MUST be placed inside the "
            "`attributes` dict. The env's reference unpacker only reads "
            "`(id, type, attributes)` — any field placed at the top level "
            "of the ref object is silently dropped."
        )
        lines.append("")
        lines.append("Attributes the env reads by name (others pass through):")
        lines.append("")
        lines.append("| `attributes.<key>` | Type | Notes |")
        lines.append("|---|---|---|")
        for fname in spec.promoted_attrs:
            finfo = fields.get(fname)
            if finfo:
                type_str = _type_label(finfo.annotation)
                lines.append(f"| `attributes.{fname}` | {type_str} | |")
        lines.append("")

    if spec.operating_tools:
        tools_str = ", ".join(f"`{t}`" for t in spec.operating_tools)
        lines.append(f"**Consumed by**: {tools_str}.")
    if spec.discovery_tools:
        disc_str = ", ".join(f"`{t}`" for t in spec.discovery_tools)
        lines.append(f"**Discoverable via**: {disc_str}.")
    lines.append("")


def _render_derived_ref_type(spec, lines: list[str]) -> None:
    lines.append(
        f"#### derived: {spec.name} (from `{spec.source_ref_type}` ref's "
        f"`{spec.source_attr}`)"
    )
    lines.append("")
    if spec.operating_tools:
        tools_str = ", ".join(f"`{t}`" for t in spec.operating_tools)
        lines.append(f"**Consumed by**: {tools_str}.")
    lines.append(
        f"When a `{spec.source_ref_type}` ref carries `{spec.source_attr}`, "
        f"the env hydrates a {spec.name} row."
    )
    lines.append("")


def _type_label(annotation) -> str:
    """Best-effort type label from a Pydantic field annotation."""
    if annotation is None:
        return "any"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is list:
        if args:
            return f"list[{_type_label(args[0])}]"
        return "list"
    if origin is dict:
        return "dict"
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        if len(non_none) == 1:
            label = _type_label(non_none[0])
            return f"{label}?" if has_none else label
    if origin is typing.Literal:
        return " / ".join(repr(a) for a in args)
    if isinstance(annotation, type):
        return annotation.__name__
    return "any"


# ---------------------------------------------------------------------------
# Prompt IO
# ---------------------------------------------------------------------------


def load_prompt(stage_dir: Path, name: str = "prompt") -> str:
    """Load `<stage_dir>/<name>.md`. Default name is 'prompt' so stages
    can just call `load_prompt(HERE)` where `HERE = Path(__file__).parent`.
    """
    path = stage_dir / f"{name}.md"
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, **kwargs: Any) -> str:
    """Replace {{KEY}} placeholders with stringified values. Objects get
    JSON-dumped with indent=2 for readability.
    """
    out = template
    for key, value in kwargs.items():
        placeholder = "{{" + key + "}}"
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        out = out.replace(placeholder, value)
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def write_json(path: Path, data: Any) -> Path:
    """Atomic write: serialize to a sibling `.tmp` file, then `os.replace` onto
    target. On POSIX, `os.replace` (== `Path.replace`) is atomic on the same
    filesystem. A SIGINT mid-serialize leaves the `.tmp` orphan but never a
    truncated target — satisfies the atomic-write invariant.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def format_structured_persona(structured: dict) -> str:
    """Convert structured persona dict to formatted text for LLM prompts.

    Nested dicts become indented sections; long text becomes standalone
    paragraphs; lists become comma-separated inline values.
    """

    def _fmt(v: Any) -> str:
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else "(none)"
        return str(v) if v else ""

    lines: list[str] = []
    for key, value in structured.items():
        if key.startswith("_"):
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"{label}:")
            for k, v in value.items():
                lines.append(f"  {k.replace('_', ' ').title()}: {_fmt(v)}")
        elif isinstance(value, list):
            lines.append(f"{label}: {_fmt(value)}")
        elif isinstance(value, str) and len(value) > 80:
            lines.append(f"{label}:")
            lines.append(value)
        else:
            lines.append(f"{label}: {_fmt(value)}")
    return "\n".join(lines)


def persona_dir(persona_id: str) -> Path:
    return PERSONAS_DIR / persona_id


# ---------------------------------------------------------------------------
# Archive helper for `--force` flows
# ---------------------------------------------------------------------------

import shutil as _shutil


def archive_to_prev(path: Path) -> Path | None:
    """Move `path` to a single rolling backup slot before regen.

    Files: `skeleton.json` → `skeleton_prev.json`
    Dirs:  `learning_sessions/` → `learning_sessions_prev/`

    Empty sources are removed without archiving (so a chained
    skeleton-force → fill-force doesn't overwrite a meaningful prev
    with the empty intermediate state). Existing archive at the prev
    slot is replaced (single-slot policy).

    Returns the archived path on success, None when nothing was
    archived (source missing or empty).
    """
    if not path.exists():
        return None
    parent = path.parent
    if path.is_dir():
        bak = parent / f"{path.name}_prev"
        # empty dir → just remove, don't clobber an existing meaningful prev
        if not any(path.iterdir()):
            path.rmdir()
            return None
    elif path.is_file():
        bak = parent / f"{path.stem}_prev{path.suffix}"
        if path.stat().st_size == 0:
            path.unlink()
            return None
    else:
        return None
    if bak.exists():
        if bak.is_dir():
            _shutil.rmtree(bak)
        else:
            bak.unlink()
    path.rename(bak)
    return bak
