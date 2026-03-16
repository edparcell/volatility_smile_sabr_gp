#!/usr/bin/env python
"""Rebuild all figures for the paper.

Usage:
    python build_figures.py          # build all
    python build_figures.py smile    # build only figures matching 'smile'
"""

import importlib
import importlib.util
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def discover_scripts(pattern: str | None = None) -> list[Path]:
    """Find all fig_*.py scripts, optionally filtered by substring."""
    scripts = sorted(SCRIPTS_DIR.glob("fig_*.py"))
    if pattern:
        scripts = [s for s in scripts if pattern in s.stem]
    return scripts


def run_script(path: Path) -> None:
    """Import and run a figure script's main() function."""
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    scripts = discover_scripts(pattern)

    if not scripts:
        print(f"No scripts found{f' matching {pattern!r}' if pattern else ''}.")
        return

    print(f"Building {len(scripts)} figure(s)...\n")
    t0 = time.time()

    for script in scripts:
        print(f"▸ {script.name}")
        t1 = time.time()
        try:
            run_script(script)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
        else:
            print(f"  ✓ done ({time.time() - t1:.1f}s)")
        print()

    print(f"All done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
