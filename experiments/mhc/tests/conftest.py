"""Make the flat sibling imports in this directory work under pytest.

The runner scripts here (`run_synthetic.py`, `run_diagnostics.py`) do their own
`sys.path.insert` and import siblings flatly (`import refusal_tests as rt`). pytest
collects from the repo root, so it needs the same path entry to import `diag_common` and
`synthetic_mhc` the same way.

Also falls back to the in-repo `src/` when `routeaudit` isn't installed, so the suite
runs on a fresh clone without `pip install -e .` first. An installed package still wins.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

if importlib.util.find_spec("routeaudit") is None:
    _SRC = _HERE.parents[2] / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))
