#!/usr/bin/env python
"""Run a RoboCasa module while preserving the NGC Torch NumPy-1 ABI."""

from __future__ import annotations

import runpy
import sys

import numpy as np
import robosuite  # noqa: F401


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_robocasa_numpy1_compat.py <module> [args ...]")
    if np.__version__ != "1.26.4":
        raise RuntimeError(f"Expected NGC-compatible NumPy 1.26.4, got {np.__version__}")
    module = sys.argv.pop(1)
    actual_version = np.__version__
    np.__version__ = "2.2.5"
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        np.__version__ = actual_version


if __name__ == "__main__":
    main()
