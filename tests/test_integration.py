from src.analysis.clustering import hierarchical_clustering, mds_reduce, tsne_reduce
from src.analysis.distance_matrix import build_matrix
from src.analysis.evaluation import evaluate
from src.io.midi_loader import load_midi, load_midi_files
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.hausdorff import hausdorff_modified, hausdorff_standard
from src.processing.normalization import normalize_minmax, normalize_zscore


def test_full_pipeline_minmax(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    assert len(curves) == 2
    assert all(curve.raw_notes for curve in curves)

    normalize_minmax(curves)
    for curve in curves:
        assert curve.points is not None
        assert curve.points.shape == (len(curve.raw_notes), 3)

    matrix = build_matrix(curves, method="standard")
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    assert matrix[0, 1] > 0.0

    modified = build_matrix(curves, method="modified")
    assert modified[0, 1] > 0.0

    frechet = build_matrix(curves, method="frechet")
    assert frechet[0, 1] > 0.0

    result = hierarchical_clustering(matrix, [curve.name for curve in curves])
    assert len(result["labels"]) == 2

    coords = mds_reduce(matrix)
    assert coords.shape == (2, 2)


def test_full_pipeline_zscore(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])

    normalize_zscore(curves)
    for curve in curves:
        assert curve.points is not None
        assert curve.points.shape == (len(curve.raw_notes), 3)

    matrix = build_matrix(curves, method="modified")
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] >= 0.0


def test_pipeline_with_labels(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    curves[0].label = "classical"
    curves[1].label = "pop"

    normalize_minmax(curves)
    matrix = build_matrix(curves, method="standard")
    result = hierarchical_clustering(matrix, [curve.name for curve in curves], n_clusters=2)

    metrics = evaluate(curves, result["labels"])
    assert "ari" in metrics
    assert "purity" in metrics


def test_pipeline_without_labels(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    normalize_minmax(curves)
    matrix = build_matrix(curves)
    result = hierarchical_clustering(matrix, [curve.name for curve in curves])

    assert evaluate(curves, result["labels"]) == {}


def test_single_file_pipeline(simple_midi_file):
    curves = load_midi_files([simple_midi_file])
    normalize_minmax(curves)

    assert curves[0].points is not None
    matrix = build_matrix(curves)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0

    result = hierarchical_clustering(matrix, [curves[0].name])
    assert result == {"linkage": None, "labels": [0]}


def test_empty_pipeline():
    curves = load_midi_files([])
    normalize_minmax(curves)

    matrix = build_matrix(curves)

    assert matrix.shape == (0, 0)


def test_non_midi_file(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("C D E F G", encoding="utf-8")

    assert load_midi(str(bad_file)) is None


def test_direct_distance_functions_after_normalization():
    left = MelodyCurve(
        name="left",
        filepath="/fake/left.mid",
        raw_notes=[Note(0.0, 60, 80), Note(1.0, 64, 90)],
    )
    right = MelodyCurve(
        name="right",
        filepath="/fake/right.mid",
        raw_notes=[Note(0.0, 62, 70), Note(1.0, 65, 95)],
    )
    curves = [left, right]

    normalize_minmax(curves)

    assert hausdorff_standard(left.points, right.points) > 0.0
    assert hausdorff_modified(left.points, right.points) > 0.0


def test_tsne_reduce_small_matrix():
    curves = [
        MelodyCurve(name="a", filepath="/fake/a.mid", raw_notes=[Note(0.0, 60, 80)]),
        MelodyCurve(name="b", filepath="/fake/b.mid", raw_notes=[Note(0.0, 62, 80)]),
        MelodyCurve(name="c", filepath="/fake/c.mid", raw_notes=[Note(0.0, 72, 80)]),
    ]
    normalize_minmax(curves)
    matrix = build_matrix(curves)

    coords = tsne_reduce(matrix)

    assert coords.shape == (3, 2)
