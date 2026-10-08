"""Pure-path and parser coverage for the tracker-output visualisation CLI."""

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from _runner import run_tests

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                     "scripts", "visualize.py")
_SPEC = importlib.util.spec_from_file_location("visualize", _PATH)
visualize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(visualize)


def test_canonical_sequence_name_accepts_short_and_full_ids():
    assert visualize.canonical_sequence_name("4") == "MOT17-04-FRCNN"
    assert visualize.canonical_sequence_name("MOT17-04-FRCNN") == "MOT17-04-FRCNN"


def test_resolvers_find_standard_pipeline_layout():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        sequence = root / "MOT17" / "train" / "MOT17-04-FRCNN"
        (sequence / "img1").mkdir(parents=True)
        results = root / "results" / "occlusion"
        results.mkdir(parents=True)
        result_file = results / "MOT17-04-FRCNN.txt"
        result_file.write_text("1,7,1,2,3,4,1,-1,-1\n")

        assert visualize.resolve_sequence_dir(root / "MOT17", "04") == sequence
        assert visualize.resolve_results_file(None, root / "results", "occlusion",
                                              sequence.name) == result_file


def test_read_tracks_keeps_valid_positive_boxes():
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "tracks.txt"
        path.write_text("1,5,10,20,30,40,1,-1,-1\n2,6,0,0,0,4,1,-1,-1\n")
        tracks = visualize.read_tracks(path)
        assert tracks[1] == [(5, 10.0, 20.0, 30.0, 40.0)]
        assert 2 not in tracks


if __name__ == "__main__":
    raise SystemExit(run_tests(dict(globals()), "visualize"))
