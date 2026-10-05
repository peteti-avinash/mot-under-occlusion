"""Generate the markdown results tables for WRITEUP.md from metrics.json.

The numbers in the write-up should never be typed by hand: transcription
errors in a results table are indistinguishable from real findings. This reads
the metrics the pipeline wrote and emits markdown, including the SORT vs
occlusion delta with its sign made explicit.

    PYTHONPATH=src ./venv/bin/python scripts/make_report.py \
        --sort results/full_sort/sort/metrics.json \
        --occlusion results/full_occlusion/occlusion/metrics.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

COLUMNS = ("mota", "motp", "hota", "idf1", "num_switches",
           "num_fragmentations", "mostly_tracked", "mostly_lost",
           "recall", "precision", "num_matches")


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _index(payload):
    return {row["sequence"]: row for row in payload["sequences"]}


def _fmt(key, value):
    if key in ("idf1",):
        return f"{value:.3f}"
    if key in ("mostly_tracked", "mostly_lost"):
        return f"{value:.1f}"
    if key in ("num_switches", "num_fragmentations", "num_matches"):
        return f"{int(round(value))}"
    return f"{value:.2f}"


def per_sequence_table(sort_rows, occ_rows):
    header = ("| sequence | MOTA | MOTP | HOTA | IDF1 | IDS | Frag | MT | ML "
              "| Rec | Prec |")
    rule = "| --- " * 11 + "|"
    lines = [header, rule]
    for seq in sorted(set(sort_rows) | set(occ_rows)):
        cells = []
        for key in COLUMNS:
            values = []
            for rows in (sort_rows, occ_rows):
                row = rows.get(seq)
                values.append(None if row is None else row.get(key, 0.0))
            if values[0] is None and values[1] is None:
                cells.append("—")
            else:
                shown = [_fmt(key, v) for v in values if v is not None]
                cells.append(shown[0] if len(shown) == 1
                             else " / ".join(shown))
        lines.append(f"| {seq.replace('-FRCNN', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def overall_table(sort_payload, occ_payload):
    lines = ["| metric | SORT | occlusion | Δ (occ − SORT) |",
             "| --- | --- | --- | --- |"]
    if occ_payload is None:
        occ_payload = {"aggregate": None}
    sort_agg = sort_payload["aggregate"]
    occ_agg = occ_payload.get("aggregate")
    for key in COLUMNS:
        a = sort_agg.get(key, 0.0)
        if occ_agg is None:
            lines.append(f"| {key} | {_fmt(key, a)} | — | — |")
            continue
        b = occ_agg.get(key, 0.0)
        better_is_lower = key in ("num_switches", "num_fragmentations",
                                  "mostly_lost")
        delta = b - a
        if key in ("num_switches", "num_fragmentations", "mostly_lost"):
            verdict = "better" if delta < 0 else ("worse" if delta > 0 else "=")
        else:
            verdict = "better" if delta > 0 else ("worse" if delta < 0 else "=")
        lines.append(f"| {key} | {_fmt(key, a)} | {_fmt(key, b)} | "
                     f"{delta:+.2f} ({verdict}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sort", default="results/full_sort/sort/metrics.json")
    ap.add_argument("--occlusion",
                    default="results/full_occlusion/occlusion/metrics.json")
    ap.add_argument("--section", default="both", choices=("both", "per-seq",
                                                          "overall"))
    args = ap.parse_args()

    sort_payload = _load(args.sort)
    if sort_payload is None:
        raise SystemExit(f"missing SORT metrics: {args.sort}")
    occ_payload = _load(args.occlusion)

    if args.section in ("both", "per-seq"):
        print("### Per sequence (SORT / occlusion)\n")
        print(per_sequence_table(_index(sort_payload),
                                 _index(occ_payload) if occ_payload else {}))
        print()
    if args.section in ("both", "overall"):
        print("### Overall\n")
        print(overall_table(sort_payload, occ_payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())