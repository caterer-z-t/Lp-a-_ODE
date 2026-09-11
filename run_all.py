"""
Reproduce all figures from Caterer (2026).

Run from the repository root:
    python run_all.py

Figures are written to ./figures/ at 300 DPI.
Figure 2 and Figure 3 both run Bayesian inference; each takes ~5 seconds.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "scripts/figure1.py",
    "scripts/figure2.py",
    "scripts/figure3.py",
    "scripts/figure4.py",
]

if __name__ == "__main__":
    root = Path(__file__).parent
    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"Running {script} …")
        print('='*60)
        result = subprocess.run(
            [sys.executable, str(root / script)],
            cwd=root,
        )
        if result.returncode != 0:
            print(f"ERROR: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll figures generated successfully.")
