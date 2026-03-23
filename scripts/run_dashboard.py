"""
Launch the Streamlit dashboard.

Usage
-----
    python scripts/run_dashboard.py [-- streamlit-args ...]

This script is a thin wrapper around:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Resolve project root so the script works from any CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "dashboard" / "app.py"


def main() -> None:
    """Run the Streamlit dashboard."""
    if not APP_PATH.exists():
        print(f"ERROR: Dashboard app not found at {APP_PATH}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)] + sys.argv[1:]
    print(f"Launching: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
