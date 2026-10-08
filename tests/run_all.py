"""Run every test suite in this directory and exit non-zero if any fails.

    PYTHONPATH=src ./venv/bin/python tests/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = sys.executable

SUITES = ("test_tracker.py", "test_detector.py", "test_metrics.py",
          "test_aggregate.py", "test_detector_ceiling.py", "test_visualize.py")


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    failed = []
    for suite in SUITES:
        print(f"\n===== {suite} =====", flush=True)
        result = subprocess.run([PYTHON, os.path.join(HERE, suite)],
                                cwd=ROOT, env=env)
        if result.returncode != 0:
            failed.append(suite)
    print("\n" + "=" * 40)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(SUITES)} suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
