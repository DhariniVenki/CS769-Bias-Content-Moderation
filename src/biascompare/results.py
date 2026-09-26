"""Serialising the results dict to disk. See DESIGN.md §7.

`stats.py` deliberately returns `float('nan')` instead of raising on
degenerate input (DESIGN §6), so the results dict built by `compare.py`
always contains plain Python/NumPy floats — some of them non-finite.
`save_results` sanitises those to JSON `null` before writing, since raw
`NaN`/`Infinity` are not valid JSON (Python's `json` module would otherwise
emit the non-standard literals `NaN`/`Infinity` silently).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def save_results(results: dict, path: str | Path) -> Path:
    """Write `results` as indented JSON to `path`, creating parent
    directories as needed. NaN/inf values become `null`. Returns `path`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize(results)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, indent=2, allow_nan=False)
    return target


def _sanitize(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value
