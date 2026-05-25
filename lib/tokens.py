"""Per-persona datagen token accounting.

Each datagen stage builds a per-stage `{model: {...}}` bucket via
`lib.llm.model_scope`, then calls `record_stage(persona_dir, stage_name,
bucket)` to merge into `data/personas/<pid>/token_usage.json`. The on-disk
file keeps three views recomputed every write:

  by_stage  {stage: {model: {prompt_tokens, completion_tokens,
                              total_tokens, calls}}}
  by_model  {model: {prompt_tokens, completion_tokens,
                     total_tokens, calls}}
  total     {prompt_tokens, completion_tokens, total_tokens, calls}

`by_stage` is the source of truth; `by_model` and `total` are derived. A
re-merge of the same stage adds onto the existing entry (so re-running a
stage with `--force` accumulates across runs by design — when you want a
fresh count, delete `token_usage.json` first).

File access is fcntl-locked so concurrent stages writing to the same
persona don't lose updates.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "calls")


def _empty_metric() -> dict:
    return {k: 0 for k in _FIELDS}


def _add_into(dst: dict, src: dict) -> None:
    """In-place: dst += src for the four counter fields."""
    for k in _FIELDS:
        dst[k] = int(dst.get(k, 0) or 0) + int(src.get(k, 0) or 0)


def _recompute_views(by_stage: dict[str, dict[str, dict]]) -> tuple[dict, dict]:
    """Derive (by_model, total) from by_stage."""
    by_model: dict[str, dict] = {}
    total = _empty_metric()
    for _stage, models in by_stage.items():
        for model, metric in models.items():
            slot = by_model.setdefault(model, _empty_metric())
            _add_into(slot, metric)
            _add_into(total, metric)
    return by_model, total


def record_stage(
    persona_dir: Path,
    stage: str,
    stage_bucket: dict[str, dict[str, int]],
) -> Path | None:
    """Merge `stage_bucket` ({model: {...}}) into the persona's token_usage.json.

    No-op when stage_bucket is empty (avoids creating a file on a stage
    that did nothing — e.g. a `--skipped` re-run).

    Returns the file path on write, None on no-op.
    """
    if not stage_bucket:
        return None
    persona_dir = Path(persona_dir)
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "token_usage.json"

    # Open + flock for read-modify-write atomicity. Use 'a+' so the file is
    # created if missing without truncating; rewind to read full contents.
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        try:
            existing: dict[str, Any] = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            existing = {}

        by_stage: dict[str, dict[str, dict]] = existing.get("by_stage", {})

        # Merge stage_bucket into by_stage[stage]
        stage_slot = by_stage.setdefault(stage, {})
        for model, metric in stage_bucket.items():
            slot = stage_slot.setdefault(model, _empty_metric())
            _add_into(slot, metric)

        by_model, total = _recompute_views(by_stage)
        out = {
            "persona_id": persona_dir.name,
            "by_stage": by_stage,
            "by_model": by_model,
            "total": total,
        }

        # Truncate + rewrite
        f.seek(0)
        f.truncate()
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
        f.flush()
    return path


def make_bucket() -> dict[str, dict[str, int]]:
    """Convenience: empty `{model: {...}}` bucket for a stage's `model_scope`."""
    return {}
