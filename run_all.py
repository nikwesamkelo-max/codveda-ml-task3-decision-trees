"""Run the full decision tree pipeline (steps 1-4) in order."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "iris.csv"
STEPS = [
    "src/01_train_decision_tree.py",
    "src/02_visualize_tree.py",
    "src/03_prune_tree.py",
    "src/04_evaluate_model.py",
]


def main():
    if not RAW_FILE.exists():
        sys.exit(f"Dataset not found: {RAW_FILE}\nPlace iris.csv there first.")

    total_start = time.time()
    for number, step in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}\nSTEP {number}/{len(STEPS)}: {step}\n{'=' * 60}")
        start = time.time()
        result = subprocess.run([sys.executable, step], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStep failed: {step}")
        print(f"Finished in {time.time() - start:.1f}s")

    print(f"\nPipeline complete in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
