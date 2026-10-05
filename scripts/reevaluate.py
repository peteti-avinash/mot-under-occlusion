"""Recompute metrics from existing MOT-format result files.

Metrics are a pure function of the hypothesis boxes and the ground truth, so
they never need a GPU or a detector run. This makes it possible to correct a
bug in src/mot/metrics.py and regenerate every reported number from the .txt
files that are already on disk, instead of rerunning inference and silently
mixing two versions of the tracker.

    PYTHONPATH=src ./venv/bin/python scripts/reevaluate.py \
        --results results/full_sort/sort --tracker sort
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from mot.dataset import find_dataset_root, list_sequences, read_boxes  # noqa: E402
from mot.metrics import aggregate, evaluate, format_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="directory holding MOT17-XX-*.txt result files")
    ap.add_argument("--tracker", default=None,
                    help="output subdirectory name; defaults to the directory name")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--write", action="store_true",
                    help="overwrite metrics.json in --results (default: dry run)")
    args = ap.parse_args()

    root = find_dataset_root(args.dataset_root)
    sequences = {s.name: s for s in list_sequences(root, detector="FRCNN")}

    result_files = sorted(f for f in os.listdir(args.results)
                          if f.startswith("MOT17-") and f.endswith(".txt"))
    if not result_files:
        raise SystemExit(f"no MOT17-*.txt files found in {args.results}")

    results = []
    for name in result_files:
        sequence = name[:-len(".txt")]
        if sequence not in sequences:
            print(f"  {sequence}: SKIPPED (not in {root}/train FRCNN)")
            continue
        seq = sequences[sequence]
        hyp = read_boxes(os.path.join(args.results, name))
        metrics = evaluate(hyp, seq.gt_file, sequence=seq.name, max_frame=seq.length)
        print(format_table([metrics], include_overall=False))
        results.append(metrics)

    if not results:
        raise SystemExit("nothing evaluated")

    overall = aggregate(results)
    print("\n=== OVERALL ===")
    print(format_table([overall], include_overall=False))

    if not args.write:
        print("\ndry run; pass --write to update metrics.json")
        return 0

    path = os.path.join(args.results, "metrics.json")
    config = {}
    if os.path.exists(path):
        with open(path) as fh:
            config = json.load(fh).get("config", {})
    config["tracker"] = args.tracker or os.path.basename(args.results.rstrip("/"))
    config["reevaluated_from_txt"] = True
    save_json(results, path, config=config)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())