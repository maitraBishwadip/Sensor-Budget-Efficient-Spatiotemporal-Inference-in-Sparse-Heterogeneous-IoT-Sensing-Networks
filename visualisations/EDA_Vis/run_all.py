"""Run every EDA visualization script in order.

Usage:
    python visualisations/EDA_Vis/run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
for script in scripts:
    print(f"\n=== Running {script.name} ===")
    t0 = time.time()
    try:
        runpy.run_path(str(script), run_name="__main__")
        print(f"  ok in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
