# 音乐旋律线几何相似性分析工具 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 desktop app that loads MIDI files, renders melody lines as 3D curves, computes pairwise Hausdorff distances, and performs clustering with evaluation.

**Architecture:** Three-layer design — Data Layer (dataclasses + numpy arrays), Core Layer (MIDI parsing, normalization, Hausdorff, clustering, evaluation), UI Layer (PySide6 main window with PyQtGraph 3D view, matplotlib-embedded panels, QTableWidget matrix).

**Tech Stack:** Python 3.11+, PySide6, pyqtgraph + PyOpenGL, music21, numpy, scipy, scikit-learn, matplotlib, pytest

---

### Task 1: Project Scaffold and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/io/__init__.py`
- Create: `src/processing/__init__.py`
- Create: `src/analysis/__init__.py`
- Create: `src/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `setup.cfg`

- [ ] **Step 1: Write `requirements.txt`**

```
PySide6>=6.5.0
pyqtgraph>=0.13.0
PyOpenGL>=3.1.7
music21>=9.1.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pytest>=7.4.0
```

- [ ] **Step 2: Write `setup.cfg`**

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
import tempfile
import os
import pytest
from music21 import note, stream, tempo, dynamics, metadata, midi


def _create_midi_file(filepath, notes_data, bpm=120):
    """Helper to write a MIDI file from note tuples: (pitch, start_beat, duration_beats, velocity)."""
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    p = stream.Part()
    for pitch_val, start, dur, vel in notes_data:
        n = note.Note(pitch_val)
        n.duration.quarterLength = dur
        n.volume.velocity = vel
        p.insert(start, n)
    s.insert(0, p)
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filepath, "wb")
    mf.write()
    mf.close()


@pytest.fixture
def simple_midi_file():
    """A simple 8-note ascending scale MIDI: C4..C5, quarter notes."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "simple_scale.mid")
    notes = []
    for i in range(8):
        notes.append((60 + i, float(i), 1.0, 80))
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)


@pytest.fixture
def two_note_midi_file():
    """Minimal MIDI with two notes for edge case testing."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "two_notes.mid")
    notes = [(60, 0.0, 1.0, 100), (64, 1.0, 1.0, 90)]
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)


@pytest.fixture
def sibling_midi_file():
    """A second MIDI with similar but shifted melody for distance testing."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "shifted.mid")
    notes = []
    for i in range(8):
        notes.append((62 + i, float(i), 1.0, 80))
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)
```

- [ ] **Step 4: Write all `__init__.py` files (empty)**

All `__init__.py` files are empty. Create them at:
- `src/__init__.py`
- `src/models/__init__.py`
- `src/io/__init__.py`
- `src/processing/__init__.py`
- `src/analysis/__init__.py`
- `src/ui/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt setup.cfg src/ tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffold with dependencies and test fixtures"
```

---

### Task 2: Data Models (Note, MelodyCurve)

**Files:**
- Create: `src/models/note.py`
- Create: `src/models/melody_curve.py`

- [ ] **Step 1: Write `src/models/note.py`**

```python
from dataclasses import dataclass


@dataclass
class Note:
    timestamp: float  # seconds from start
    pitch: int        # MIDI note number 0-127
    velocity: int     # MIDI velocity 0-127

    def as_tuple(self):
        return (self.timestamp, self.pitch, self.velocity)
```

- [ ] **Step 2: Write `src/models/melody_curve.py`**

```python
from dataclasses import dataclass, field
import numpy as np
from .note import Note


@dataclass
class MelodyCurve:
    name: str
    filepath: str
    label: str | None = None
    raw_notes: list[Note] = field(default_factory=list)
    points: np.ndarray | None = None  # shape (N, 3) normalized
    color: str = "#ffffff"

    @property
    def num_notes(self) -> int:
        return len(self.raw_notes)
```

- [ ] **Step 3: Commit**

```bash
git add src/models/
git commit -m "feat: add Note and MelodyCurve data models"
```

---

### Task 3: MIDI Loader

**Files:**
- Create: `src/io/midi_loader.py`
- Create: `tests/test_midi_loader.py`
- Modify: `src/models/note.py` (already exists, no changes)

- [ ] **Step 1: Write failing test `tests/test_midi_loader.py`**

```python
import os
from src.io.midi_loader import load_midi, load_midi_files


def test_load_midi_returns_melody_curve(simple_midi_file):
    curve = load_midi(simple_midi_file)
    assert curve is not None
    assert curve.name == "simple_scale"
    assert curve.filepath == simple_midi_file
    assert len(curve.raw_notes) == 8
    assert curve.raw_notes[0].pitch == 60
    assert curve.raw_notes[0].timestamp == 0.0
    assert curve.raw_notes[0].velocity == 80


def test_load_midi_sets_label_none(simple_midi_file):
    curve = load_midi(simple_midi_file)
    assert curve.label is None


def test_load_midi_files_batch(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    assert len(curves) == 2
    assert curves[0].name == "simple_scale"
    assert curves[1].name == "shifted"


def test_load_midi_rejects_non_midi(tmp_path):
    bad_file = tmp_path / "not_midi.txt"
    bad_file.write_text("hello")
    curve = load_midi(str(bad_file))
    assert curve is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_midi_loader.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/io/midi_loader.py`**

```python
import os
from src.models.melody_curve import MelodyCurve
from src.models.note import Note


def load_midi(filepath: str) -> MelodyCurve | None:
    """Parse a .mid file into a MelodyCurve. Returns None on failure."""
    if not filepath.lower().endswith((".mid", ".midi")):
        return None

    try:
        from music21 import converter
        score = converter.parse(filepath)
    except Exception:
        return None

    name = os.path.splitext(os.path.basename(filepath))[0]

    notes = []
    for part in score.parts:
        for element in part.flat.notes:
            pitch_val = element.pitch.midi if element.pitch else 0
            offset = float(element.offset)
            velocity_val = element.volume.velocity if element.volume.velocity else 0
            notes.append(Note(timestamp=offset, pitch=pitch_val, velocity=velocity_val))

    if not notes:
        return None

    notes.sort(key=lambda n: n.timestamp)

    return MelodyCurve(
        name=name,
        filepath=filepath,
        raw_notes=notes,
    )


def load_midi_files(filepaths: list[str]) -> list[MelodyCurve]:
    """Load multiple MIDI files, skipping failures."""
    curves = []
    for fp in filepaths:
        curve = load_midi(fp)
        if curve is not None:
            curves.append(curve)
    return curves
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_midi_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/io/midi_loader.py tests/test_midi_loader.py
git commit -m "feat: implement MIDI loader with music21"
```

---

### Task 4: Normalization (Min-Max and Z-Score)

**Files:**
- Create: `src/processing/normalization.py`
- Create: `tests/test_normalization.py`

- [ ] **Step 1: Write failing test `tests/test_normalization.py`**

```python
import numpy as np
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.normalization import normalize_minmax, normalize_zscore


def make_curve(name, notes_data):
    """notes_data: list of (time, pitch, velocity)"""
    raw = [Note(t, p, v) for t, p, v in notes_data]
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", raw_notes=raw)


def test_normalize_minmax_maps_to_unit_cube():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 60)])
    c2 = make_curve("b", [(0.0, 50, 50), (3.0, 72, 127)])

    normalize_minmax([c1, c2])

    # After normalization, all points should be in [0, 1]^3
    for c in [c1, c2]:
        assert c.points is not None
        assert np.all(c.points >= 0.0)
        assert np.all(c.points <= 1.0)


def test_normalize_minmax_same_curve_zero_variance():
    c1 = make_curve("single", [(1.0, 60, 80)])
    normalize_minmax([c1])
    # Single point: min==max for time, so time becomes 0; pitch and velocity also
    assert c1.points is not None
    assert c1.points.shape == (1, 3)


def test_normalize_zscore_maps_to_zero_mean():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 60)])
    normalize_zscore([c1])
    assert c1.points is not None
    means = np.mean(c1.points, axis=0)
    assert np.allclose(means, 0.0, atol=1e-10)


def test_normalize_preserves_relative_structure():
    c1 = make_curve("a", [(0.0, 60, 80), (2.0, 72, 100)])
    c2 = make_curve("b", [(0.0, 64, 90), (1.0, 67, 80)])
    normalize_minmax([c1, c2])

    # After normalization, c1 time span should be larger than c2 time span
    span1 = np.max(c1.points[:, 0]) - np.min(c1.points[:, 0])
    span2 = np.max(c2.points[:, 0]) - np.min(c2.points[:, 0])
    assert span1 > span2


def test_normalize_empty_curves_list():
    normalize_minmax([])
    normalize_zscore([])
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_normalization.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/processing/normalization.py`**

```python
import numpy as np
from src.models.melody_curve import MelodyCurve


def normalize_minmax(curves: list[MelodyCurve]) -> None:
    """Normalize all curves to [0, 1]^3 using global min-max. Mutates .points."""
    if not curves:
        return

    # Collect all raw points for global min/max
    all_t = []
    all_p = []
    all_v = []
    for c in curves:
        for n in c.raw_notes:
            all_t.append(n.timestamp)
            all_p.append(n.pitch)
            all_v.append(n.velocity)

    t_arr = np.array(all_t)
    p_arr = np.array(all_p)
    v_arr = np.array(all_v)

    t_min, t_max = t_arr.min(), t_arr.max()
    p_min, p_max = p_arr.min(), p_arr.max()
    v_min, v_max = v_arr.min(), v_arr.max()

    t_range = t_max - t_min
    p_range = p_max - p_min
    v_range = v_max - v_min

    for c in curves:
        pts = np.array([(n.timestamp, n.pitch, n.velocity) for n in c.raw_notes], dtype=np.float64)
        if len(pts) == 0:
            c.points = pts
            continue
        pts[:, 0] = (pts[:, 0] - t_min) / t_range if t_range > 0 else 0.0
        pts[:, 1] = (pts[:, 1] - p_min) / p_range if p_range > 0 else 0.5
        pts[:, 2] = (pts[:, 2] - v_min) / v_range if v_range > 0 else 0.5
        c.points = pts


def normalize_zscore(curves: list[MelodyCurve]) -> None:
    """Normalize all curves to zero mean, unit std per dimension. Mutates .points."""
    if not curves:
        return

    all_t = []
    all_p = []
    all_v = []
    for c in curves:
        for n in c.raw_notes:
            all_t.append(n.timestamp)
            all_p.append(n.pitch)
            all_v.append(n.velocity)

    t_arr = np.array(all_t)
    p_arr = np.array(all_p)
    v_arr = np.array(all_v)

    t_mean, t_std = t_arr.mean(), t_arr.std()
    p_mean, p_std = p_arr.mean(), p_arr.std()
    v_mean, v_std = v_arr.mean(), v_arr.std()

    for c in curves:
        pts = np.array([(n.timestamp, n.pitch, n.velocity) for n in c.raw_notes], dtype=np.float64)
        if len(pts) == 0:
            c.points = pts
            continue
        pts[:, 0] = (pts[:, 0] - t_mean) / t_std if t_std > 0 else 0.0
        pts[:, 1] = (pts[:, 1] - p_mean) / p_std if p_std > 0 else 0.0
        pts[:, 2] = (pts[:, 2] - v_mean) / v_std if v_std > 0 else 0.0
        c.points = pts
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_normalization.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/processing/normalization.py tests/test_normalization.py
git commit -m "feat: implement Min-Max and Z-Score normalization"
```

---

### Task 5: Hausdorff Distance (Standard + Modified)

**Files:**
- Create: `src/processing/hausdorff.py`
- Create: `tests/test_hausdorff.py`

- [ ] **Step 1: Write failing test `tests/test_hausdorff.py`**

```python
import numpy as np
from src.processing.hausdorff import hausdorff_standard, hausdorff_modified


def test_hausdorff_identical_sets():
    A = np.array([[0.0, 0.5, 0.3], [1.0, 0.8, 0.2], [0.5, 0.3, 0.9]])
    B = A.copy()
    assert hausdorff_standard(A, B) == pytest.approx(0.0, abs=1e-10)
    assert hausdorff_modified(A, B) == pytest.approx(0.0, abs=1e-10)


def test_hausdorff_different_sets_positive():
    A = np.array([[0.0, 0.0, 0.0]])
    B = np.array([[1.0, 1.0, 1.0]])
    dist = hausdorff_standard(A, B)
    assert dist > 0.0


def test_hausdorff_modified_less_than_standard():
    np.random.seed(42)
    A = np.random.rand(20, 3)
    B = np.random.rand(20, 3)
    # Add an outlier to B
    B[-1] = np.array([5.0, 5.0, 5.0])
    h_std = hausdorff_standard(A, B)
    h_mod = hausdorff_modified(A, B)
    assert h_mod < h_std


def test_hausdorff_symmetry():
    A = np.array([[0.0, 0.5, 0.3], [1.0, 0.8, 0.2]])
    B = np.array([[0.2, 0.3, 0.7], [0.9, 0.6, 0.1]])
    assert hausdorff_standard(A, B) == pytest.approx(hausdorff_standard(B, A))
    assert hausdorff_modified(A, B) == pytest.approx(hausdorff_modified(B, A))


def test_hausdorff_single_point_each():
    A = np.array([[0.0, 0.0, 0.0]])
    B = np.array([[0.3, 0.4, 0.0]])
    dist = hausdorff_standard(A, B)
    assert dist == pytest.approx(0.5)


import pytest
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_hausdorff.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/processing/hausdorff.py`**

```python
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree


def hausdorff_standard(A: np.ndarray, B: np.ndarray) -> float:
    """Standard Hausdorff distance between two point sets."""
    h_AB = directed_hausdorff(A, B)[0]
    h_BA = directed_hausdorff(B, A)[0]
    return max(h_AB, h_BA)


def hausdorff_modified(A: np.ndarray, B: np.ndarray) -> float:
    """Modified Hausdorff distance using mean instead of max."""
    tree_B = KDTree(B)
    d_AB, _ = tree_B.query(A)
    h_AB = float(np.mean(d_AB))

    tree_A = KDTree(A)
    d_BA, _ = tree_A.query(B)
    h_BA = float(np.mean(d_BA))

    return max(h_AB, h_BA)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_hausdorff.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/processing/hausdorff.py tests/test_hausdorff.py
git commit -m "feat: implement standard and modified Hausdorff distance"
```

---

### Task 6: Distance Matrix Builder

**Files:**
- Create: `src/analysis/distance_matrix.py`
- Create: `tests/test_distance_matrix.py`

- [ ] **Step 1: Write failing test `tests/test_distance_matrix.py`**

```python
import numpy as np
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.normalization import normalize_minmax
from src.analysis.distance_matrix import build_matrix


def make_curve(name, notes_data):
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(t, p, v) for t, p, v in notes_data],
    )


def test_build_matrix_square_symmetric():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    c2 = make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)])
    c3 = make_curve("c", [(0.0, 72, 110), (0.5, 75, 120)])
    curves = [c1, c2, c3]
    normalize_minmax(curves)

    matrix = build_matrix(curves, method="standard")
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[1, 1] == pytest.approx(0.0)
    assert matrix[2, 2] == pytest.approx(0.0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)


def test_build_matrix_modified_method():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    c2 = make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)])
    curves = [c1, c2]
    normalize_minmax(curves)

    matrix = build_matrix(curves, method="modified")
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.0


def test_build_matrix_single_curve():
    c1 = make_curve("solo", [(0.0, 60, 80)])
    normalize_minmax([c1])
    matrix = build_matrix([c1], method="standard")
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0


def test_build_matrix_empty_list():
    matrix = build_matrix([], method="standard")
    assert matrix.shape == (0, 0)


import pytest
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_distance_matrix.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/analysis/distance_matrix.py`**

```python
import numpy as np
from src.models.melody_curve import MelodyCurve
from src.processing.hausdorff import hausdorff_standard, hausdorff_modified


def build_matrix(curves: list[MelodyCurve], method: str = "standard") -> np.ndarray:
    """Compute NxN pairwise Hausdorff distance matrix."""
    n = len(curves)
    if n == 0:
        return np.zeros((0, 0))

    distance_fn = hausdorff_standard if method == "standard" else hausdorff_modified

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if curves[i].points is None or curves[j].points is None:
                continue
            if len(curves[i].points) == 0 or len(curves[j].points) == 0:
                continue
            d = distance_fn(curves[i].points, curves[j].points)
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_distance_matrix.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/distance_matrix.py tests/test_distance_matrix.py
git commit -m "feat: implement pairwise distance matrix builder"
```

---

### Task 7: Clustering and Dimensionality Reduction

**Files:**
- Create: `src/analysis/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test `tests/test_clustering.py`**

```python
import numpy as np
from src.analysis.clustering import hierarchical_clustering, mds_reduce, tsne_reduce


def test_hierarchical_clustering_returns_linkage_and_labels():
    matrix = np.array([
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.0, 0.7, 0.8],
        [0.8, 0.7, 0.0, 0.2],
        [0.9, 0.8, 0.2, 0.0],
    ])
    names = ["a", "b", "c", "d"]
    result = hierarchical_clustering(matrix, names)
    assert "linkage" in result
    assert "labels" in result
    assert len(result["labels"]) == 4
    # a and b should be in same cluster
    assert result["labels"][0] == result["labels"][1]


def test_mds_reduce_2d():
    matrix = np.array([
        [0.0, 0.1, 0.8],
        [0.1, 0.0, 0.7],
        [0.8, 0.7, 0.0],
    ])
    coords = mds_reduce(matrix, n_components=2)
    assert coords.shape == (3, 2)


def test_tsne_reduce_2d():
    matrix = np.array([
        [0.0, 0.1, 0.8],
        [0.1, 0.0, 0.7],
        [0.8, 0.7, 0.0],
    ])
    coords = tsne_reduce(matrix, n_components=2)
    assert coords.shape == (3, 2)


def test_hierarchical_clustering_two_clusters():
    matrix = np.array([
        [0.0, 0.05],
        [0.05, 0.0],
    ])
    names = ["x", "y"]
    result = hierarchical_clustering(matrix, names, n_clusters=2)
    assert "linkage" in result
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_clustering.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/analysis/clustering.py`**

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS, TSNE


def hierarchical_clustering(matrix: np.ndarray, names: list[str], n_clusters: int = 2) -> dict:
    """Perform agglomerative hierarchical clustering on distance matrix.

    Returns dict with:
        - linkage: scipy linkage matrix
        - labels: list of cluster labels (int) for each curve
    """
    n = len(names)
    if n < 2:
        return {"linkage": None, "labels": [0] * n}

    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(matrix[i, j])

    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    return {
        "linkage": Z,
        "labels": [int(lb) for lb in labels],
    }


def mds_reduce(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce distance matrix to 2D/3D coordinates using MDS."""
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], n_components))

    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    return mds.fit_transform(matrix)


def tsne_reduce(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce distance matrix to 2D/3D coordinates using t-SNE."""
    if matrix.shape[0] < 3:
        # t-SNE requires at least 3 samples
        return mds_reduce(matrix, n_components)

    tsne = TSNE(
        n_components=n_components,
        metric="precomputed",
        random_state=42,
        init="random",
    )
    return tsne.fit_transform(matrix)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_clustering.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clustering.py tests/test_clustering.py
git commit -m "feat: implement hierarchical clustering, MDS, and t-SNE"
```

---

### Task 8: Evaluation Metrics

**Files:**
- Create: `src/analysis/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing test `tests/test_evaluation.py`**

```python
from src.analysis.evaluation import evaluate
from src.models.melody_curve import MelodyCurve


def make_curve(name, label):
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", label=label)


def test_evaluate_returns_metrics_with_labels():
    curves = [
        make_curve("a", "古典"),
        make_curve("b", "古典"),
        make_curve("c", "流行"),
        make_curve("d", "流行"),
    ]
    cluster_labels = [1, 1, 2, 2]  # perfect clustering
    result = evaluate(curves, cluster_labels)
    assert "ari" in result
    assert "purity" in result
    assert result["ari"] == pytest.approx(1.0)
    assert result["purity"] == pytest.approx(1.0)


def test_evaluate_returns_empty_without_labels():
    curves = [
        make_curve("a", None),
        make_curve("b", None),
    ]
    cluster_labels = [1, 1]
    result = evaluate(curves, cluster_labels)
    assert result == {}


def test_evaluate_mixed_labels():
    curves = [
        make_curve("a", "古典"),
        make_curve("b", None),
        make_curve("c", "流行"),
    ]
    cluster_labels = [1, 1, 2]
    result = evaluate(curves, cluster_labels)
    assert result == {}


import pytest
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `src/analysis/evaluation.py`**

```python
from sklearn.metrics import adjusted_rand_score
from src.models.melody_curve import MelodyCurve


def evaluate(curves: list[MelodyCurve], cluster_labels: list[int]) -> dict:
    """Compute ARI and purity if all curves have non-None labels.

    Returns empty dict if any curve has no label or labels are all same.
    """
    true_labels = [c.label for c in curves]
    if any(lb is None for lb in true_labels):
        return {}

    unique_true = set(true_labels)
    if len(unique_true) < 2:
        return {}

    ari = adjusted_rand_score(true_labels, cluster_labels)
    purity = _compute_purity(true_labels, cluster_labels)

    return {"ari": round(ari, 4), "purity": round(purity, 4)}


def _compute_purity(true_labels: list[str], cluster_labels: list[int]) -> float:
    """Cluster purity: fraction of points in their cluster's majority class."""
    from collections import defaultdict, Counter

    clusters = defaultdict(list)
    for t, c in zip(true_labels, cluster_labels):
        clusters[c].append(t)

    total = len(true_labels)
    correct = 0
    for items in clusters.values():
        majority_count = Counter(items).most_common(1)[0][1]
        correct += majority_count

    return correct / total
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/evaluation.py tests/test_evaluation.py
git commit -m "feat: implement evaluation metrics (ARI, purity)"
```

---

### Task 9: Main Window Skeleton + Control Bar

**Files:**
- Create: `src/ui/main_window.py`
- Create: `src/ui/control_bar.py`

- [ ] **Step 1: Write `src/ui/control_bar.py`**

```python
from PySide6.QtWidgets import QWidget, QHBoxLayout, QComboBox, QPushButton, QLabel, QCheckBox
from PySide6.QtCore import Signal


class ControlBar(QWidget):
    method_changed = Signal(str)
    normalization_changed = Signal(str)
    compute_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        layout.addWidget(QLabel("距离算法:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["标准 Hausdorff", "Modified Hausdorff"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        layout.addWidget(self.method_combo)

        layout.addSpacing(16)
        layout.addWidget(QLabel("归一化:"))

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Min-Max [0,1]", "Z-Score"])
        self.norm_combo.currentTextChanged.connect(self._on_norm_changed)
        layout.addWidget(self.norm_combo)

        layout.addSpacing(16)

        self.compute_btn = QPushButton("计算距离矩阵")
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        layout.addWidget(self.compute_btn)

        self.export_btn = QPushButton("导出 CSV")
        self.export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def _on_method_changed(self, text: str):
        method = "standard" if "标准" in text else "modified"
        self.method_changed.emit(method)

    def _on_norm_changed(self, text: str):
        norm = "minmax" if "Min-Max" in text else "zscore"
        self.normalization_changed.emit(norm)

    def get_method(self) -> str:
        return "standard" if "标准" in self.method_combo.currentText() else "modified"

    def get_norm(self) -> str:
        return "minmax" if "Min-Max" in self.norm_combo.currentText() else "zscore"

    def set_status(self, text: str):
        self.status_label.setText(text)
```

- [ ] **Step 2: Write `src/ui/main_window.py`**

```python
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
)
from PySide6.QtCore import Qt

from .file_panel import FilePanel
from .gl_view import GLView
from .matrix_panel import MatrixPanel
from .cluster_panel import ClusterPanel
from .control_bar import ControlBar


TITLE = "音乐旋律线几何相似性分析工具"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1200, 800)

        self._curves = []
        self._matrix = None
        self._cluster_result = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # === Top: 3-panel splitter ===
        self.splitter = QSplitter(Qt.Horizontal)

        self.file_panel = FilePanel()
        self.splitter.addWidget(self.file_panel)

        self.gl_view = GLView()
        self.splitter.addWidget(self.gl_view)

        # Right side: matrix + cluster stacked
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.matrix_panel = MatrixPanel()
        right_layout.addWidget(self.matrix_panel)

        self.cluster_panel = ClusterPanel()
        right_layout.addWidget(self.cluster_panel)

        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([220, 500, 460])

        main_layout.addWidget(self.splitter)

        # === Bottom: control bar ===
        self.control_bar = ControlBar()
        main_layout.addWidget(self.control_bar)

        self._connect_signals()

    def _connect_signals(self):
        self.file_panel.files_loaded.connect(self._on_files_loaded)
        self.file_panel.visibility_changed.connect(self._on_visibility_changed)
        self.file_panel.label_changed.connect(self._on_label_changed)
        self.control_bar.compute_requested.connect(self._on_compute)
        self.control_bar.export_requested.connect(self._on_export)

    def _on_files_loaded(self):
        self._curves = self.file_panel.get_curves()
        self._run_normalization()
        self.gl_view.set_curves(self._curves)
        n = len(self._curves)
        self.control_bar.set_status(f"已加载 {n} 个文件")

    def _run_normalization(self):
        from src.processing.normalization import normalize_minmax, normalize_zscore
        norm = self.control_bar.get_norm()
        fn = normalize_minmax if norm == "minmax" else normalize_zscore
        fn(self._curves)

    def _on_visibility_changed(self):
        visible_idx = self.file_panel.get_visible_indices()
        visible_curves = [self._curves[i] for i in visible_idx if 0 <= i < len(self._curves)]
        self.gl_view.set_curves(visible_curves)

    def _on_label_changed(self, index: int, label: str):
        if 0 <= index < len(self._curves):
            self._curves[index].label = label if label else None

    def _on_compute(self):
        from src.analysis.distance_matrix import build_matrix
        from src.analysis.clustering import hierarchical_clustering

        if len(self._curves) < 2:
            self.control_bar.set_status("至少需要 2 条曲线才能计算距离矩阵")
            return

        self._run_normalization()

        method = self.control_bar.get_method()
        self._matrix = build_matrix(self._curves, method=method)
        self.matrix_panel.set_matrix(self._matrix, [c.name for c in self._curves])

        names = [c.name for c in self._curves]
        self._cluster_result = hierarchical_clustering(self._matrix, names)
        self.cluster_panel.set_result(self._matrix, names, self._cluster_result, self._curves)

        n = len(self._curves)
        self.control_bar.set_status(f"距离矩阵 {n}x{n} 计算完成")

    def _on_export(self):
        from PySide6.QtWidgets import QFileDialog
        if self._matrix is None:
            self.control_bar.set_status("请先计算距离矩阵")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出距离矩阵", "", "CSV Files (*.csv)"
        )
        if filepath:
            import numpy as np
            header = ",".join([c.name for c in self._curves])
            np.savetxt(filepath, self._matrix, delimiter=",", header=header, comments="", fmt="%.6f")
            self.control_bar.set_status(f"已导出到 {filepath}")
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/main_window.py src/ui/control_bar.py
git commit -m "feat: add main window skeleton and control bar"
```

---

### Task 10: File Panel

**Files:**
- Create: `src/ui/file_panel.py`

- [ ] **Step 1: Write `src/ui/file_panel.py`**

```python
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QFileDialog, QInputDialog, QMenu
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction
from src.io.midi_loader import load_midi_files


COLORS = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8",
    "#00cec9", "#fab1a0", "#81ecec", "#55efc4", "#74b9ff",
    "#e17055", "#6c5ce7", "#00b894", "#e84393", "#0984e3",
    "#fdcb6e", "#636e72", "#d63031", "#2d3436", "#b2bec3",
]


class FilePanel(QWidget):
    files_loaded = Signal()
    visibility_changed = Signal()
    label_changed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.import_btn = QPushButton("+ 导入 MIDI")
        self.import_btn.clicked.connect(self._import_files)
        layout.addWidget(self.import_btn)

        self.list_widget = QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        self._curves = []

    def _import_files(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "导入 MIDI 文件", "", "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        if not filepaths:
            return

        new_curves = load_midi_files(filepaths)
        for curve in new_curves:
            color_idx = len(self._curves) % len(COLORS)
            curve.color = COLORS[color_idx]
            self._curves.append(curve)

        self._rebuild_list()
        self.files_loaded.emit()

    def _rebuild_list(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, curve in enumerate(self._curves):
            text = curve.name
            if curve.label:
                text += f" [{curve.label}]"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            item.setForeground(Qt.GlobalColor.black if curve.color == "#ffffff" else None)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item):
        self.visibility_changed.emit()

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        idx = item.data(Qt.UserRole)
        if idx is None:
            return

        menu = QMenu(self)
        edit_action = QAction("编辑标签", self)
        edit_action.triggered.connect(lambda: self._edit_label(idx))
        menu.addAction(edit_action)

        remove_action = QAction("移除", self)
        remove_action.triggered.connect(lambda: self._remove_curve(idx))
        menu.addAction(remove_action)

        menu.exec_(self.list_widget.viewport().mapToGlobal(pos))

    def _edit_label(self, idx: int):
        current = self._curves[idx].label or ""
        label, ok = QInputDialog.getText(self, "编辑标签", "曲风标签:", text=current)
        if ok:
            self._curves[idx].label = label if label else None
            self._rebuild_list()
            self.label_changed.emit(idx, label)

    def _remove_curve(self, idx: int):
        self._curves.pop(idx)
        self._rebuild_list()
        self.files_loaded.emit()

    def get_curves(self):
        return self._curves

    def get_visible_indices(self):
        indices = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                indices.append(item.data(Qt.UserRole))
        return indices
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/file_panel.py
git commit -m "feat: implement file panel with import, labels, and context menu"
```

---

### Task 11: 3D GL View

**Files:**
- Create: `src/ui/gl_view.py`

- [ ] **Step 1: Write `src/ui/gl_view.py`**

```python
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt


try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class GLView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._gl_widget = gl.GLViewWidget()
            self._gl_widget.setCameraPosition(distance=2.5, elevation=30, azimuth=-45)

            # Coordinate axes
            gx = gl.GLGridItem()
            gx.setSize(2, 2, 2)
            gx.setSpacing(0.25, 0.25)
            gx.translate(1, 0, 0)
            self._gl_widget.addItem(gx)

            self._axis_lines = []

            layout.addWidget(self._gl_widget)
            self._line_items = []
        else:
            from PySide6.QtWidgets import QLabel
            label = QLabel("pyqtgraph 未安装，3D 视图不可用")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self._gl_widget = None

    def set_curves(self, curves):
        if self._gl_widget is None:
            return

        # Clear existing lines
        for item in self._line_items:
            self._gl_widget.removeItem(item)
        self._line_items.clear()

        for curve in curves:
            if curve.points is None or len(curve.points) == 0:
                continue

            pts = curve.points

            # Create line segments
            line_data = np.zeros((len(pts), 3), dtype=np.float32)
            line_data[:, 0] = pts[:, 0]  # time -> x
            line_data[:, 1] = pts[:, 1]  # pitch -> y
            line_data[:, 2] = pts[:, 2]  # velocity -> z

            # Color from hex
            hex_color = curve.color.lstrip("#")
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0

            line = gl.GLLinePlotItem(
                pos=line_data,
                color=(r, g, b, 1.0),
                width=2.0,
                antialias=True,
            )
            self._gl_widget.addItem(line)
            self._line_items.append(line)
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/gl_view.py
git commit -m "feat: implement 3D melody line view with PyQtGraph OpenGL"
```

---

### Task 12: Matrix Panel

**Files:**
- Create: `src/ui/matrix_panel.py`

- [ ] **Step 1: Write `src/ui/matrix_panel.py`**

```python
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class MatrixPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("距离矩阵")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        self._matrix = None
        self._names = []

    def set_matrix(self, matrix: np.ndarray, names: list[str]):
        self._matrix = matrix
        self._names = names
        n = len(names)

        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        self.table.setHorizontalHeaderLabels(names)
        self.table.setVerticalHeaderLabels(names)

        if n == 0:
            return

        max_dist = np.max(matrix) if n > 1 else 1.0
        if max_dist == 0:
            max_dist = 1.0

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignCenter)

                if i == j:
                    item.setBackground(QColor("#e8e8e8"))
                else:
                    ratio = val / max_dist
                    r = int(255 * ratio)
                    g = int(255 * (1 - ratio))
                    b = 50
                    item.setBackground(QColor(r, g, b))
                    if ratio > 0.5:
                        item.setForeground(QColor("white"))

                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def get_matrix(self):
        return self._matrix
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/matrix_panel.py
git commit -m "feat: implement distance matrix panel with heatmap coloring"
```

---

### Task 13: Cluster Panel

**Files:**
- Create: `src/ui/cluster_panel.py`

- [ ] **Step 1: Write `src/ui/cluster_panel.py`**

```python
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.cluster.hierarchy import dendrogram


class ClusterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("聚类结果")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        self._mds_tab = QWidget()
        self._mds_layout = QVBoxLayout(self._mds_tab)
        self._mds_layout.setContentsMargins(0, 0, 0, 0)
        self._mds_figure = Figure(figsize=(4, 3), dpi=100)
        self._mds_canvas = FigureCanvas(self._mds_figure)
        self._mds_layout.addWidget(self._mds_canvas)
        self.tabs.addTab(self._mds_tab, "MDS")

        self._dendro_tab = QWidget()
        self._dendro_layout = QVBoxLayout(self._dendro_tab)
        self._dendro_layout.setContentsMargins(0, 0, 0, 0)
        self._dendro_figure = Figure(figsize=(4, 3), dpi=100)
        self._dendro_canvas = FigureCanvas(self._dendro_figure)
        self._dendro_layout.addWidget(self._dendro_canvas)
        self.tabs.addTab(self._dendro_tab, "树状图")

        self._eval_tab = QWidget()
        self._eval_layout = QVBoxLayout(self._eval_tab)
        self._eval_layout.setContentsMargins(8, 8, 8, 8)
        self._eval_label = QLabel("（需要曲风标签）")
        self._eval_label.setAlignment(Qt.AlignCenter)
        self._eval_layout.addWidget(self._eval_label)
        self._eval_layout.addStretch()
        self.tabs.addTab(self._eval_tab, "评估")

        layout.addWidget(self.tabs)

    def set_result(self, matrix: np.ndarray, names: list[str], cluster_result: dict, curves: list):
        self._draw_mds(matrix, names, cluster_result, curves)
        self._draw_dendrogram(cluster_result, names)
        self._show_evaluation(curves, cluster_result)

    def _draw_mds(self, matrix, names, cluster_result, curves):
        from src.analysis.clustering import mds_reduce

        self._mds_figure.clear()
        ax = self._mds_figure.add_subplot(111)

        if matrix.shape[0] >= 2:
            coords = mds_reduce(matrix, n_components=2)

            labels_present = any(c.label is not None for c in curves)
            if labels_present:
                unique_labels = list(set(c.label for c in curves if c.label is not None))
                color_map = {}
                palette = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8",
                           "#00cec9", "#fab1a0", "#81ecec"]
                for i, lbl in enumerate(unique_labels):
                    color_map[lbl] = palette[i % len(palette)]

                for i, c in enumerate(curves):
                    clr = color_map.get(c.label, "#999999")
                    ax.scatter(coords[i, 0], coords[i, 1], c=clr, s=40, label=c.label, edgecolors="black", linewidth=0.5)
            else:
                cluster_labels = cluster_result.get("labels", [0] * len(curves))
                colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8"]
                for i in range(len(curves)):
                    cl = cluster_labels[i] if i < len(cluster_labels) else 0
                    clr = colors[cl % len(colors)]
                    ax.scatter(coords[i, 0], coords[i, 1], c=clr, s=40, edgecolors="black", linewidth=0.5)

            for i, name in enumerate(names):
                ax.annotate(name, (coords[i, 0], coords[i, 1]),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=7, ha="center")

            ax.set_xlabel("MDS 1")
            ax.set_ylabel("MDS 2")
        else:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax.transAxes)

        ax.set_title("MDS 降维散点图")
        self._mds_figure.tight_layout()
        self._mds_canvas.draw()

    def _draw_dendrogram(self, cluster_result, names):
        self._dendro_figure.clear()
        ax = self._dendro_figure.add_subplot(111)

        Z = cluster_result.get("linkage")
        if Z is not None and len(names) >= 2:
            dendrogram(Z, labels=names, ax=ax, leaf_rotation=45, leaf_font_size=8)
            ax.set_title("层次聚类树状图 (Ward)")
        else:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax.transAxes)

        self._dendro_figure.tight_layout()
        self._dendro_canvas.draw()

    def _show_evaluation(self, curves, cluster_result):
        from src.analysis.evaluation import evaluate

        labels = cluster_result.get("labels", [])
        result = evaluate(curves, labels)

        if result:
            text = f"ARI: {result['ari']:.4f}  |  纯度: {result['purity']:.4f}"
        else:
            labeled_count = sum(1 for c in curves if c.label is not None)
            if labeled_count == 0:
                text = "（无曲风标签）"
            elif labeled_count < len(curves):
                text = "（部分曲线无标签，无法评估）"
            else:
                text = "（标签种类不足，无法评估）"

        self._eval_label.setText(text)
```

- [ ] **Step 2: Commit**

```bash
git add src/ui/cluster_panel.py
git commit -m "feat: implement cluster panel with MDS, dendrogram, and evaluation"
```

---

### Task 14: Application Entry Point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Write `src/main.py`**

```python
import sys
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: add application entry point"
```

---

### Task 15: End-to-End Integration Test + Error Handling Polish

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test `tests/test_integration.py`**

```python
from src.io.midi_loader import load_midi, load_midi_files
from src.processing.normalization import normalize_minmax, normalize_zscore
from src.processing.hausdorff import hausdorff_standard, hausdorff_modified
from src.analysis.distance_matrix import build_matrix
from src.analysis.clustering import hierarchical_clustering, mds_reduce, tsne_reduce
from src.analysis.evaluation import evaluate
from src.models.melody_curve import MelodyCurve
from src.models.note import Note


def test_full_pipeline_minmax(simple_midi_file, sibling_midi_file):
    # Load
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    assert len(curves) == 2
    assert all(c.raw_notes for c in curves)

    # Normalize
    normalize_minmax(curves)
    for c in curves:
        assert c.points is not None
        assert c.points.shape[0] == len(c.raw_notes)
        assert c.points.shape[1] == 3

    # Distance matrix
    matrix = build_matrix(curves, method="standard")
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    assert matrix[0, 1] > 0.0

    # Modified
    matrix_mod = build_matrix(curves, method="modified")
    assert matrix_mod[0, 1] > 0.0

    # Clustering
    result = hierarchical_clustering(matrix, [c.name for c in curves])
    assert len(result["labels"]) == 2

    # MDS
    coords = mds_reduce(matrix)
    assert coords.shape == (2, 2)


def test_full_pipeline_zscore(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    normalize_zscore(curves)
    for c in curves:
        assert c.points is not None
        assert c.points.shape[0] == len(c.raw_notes)

    matrix = build_matrix(curves, method="modified")
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.0


def test_pipeline_with_labels(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    curves[0].label = "古典"
    curves[1].label = "流行"

    normalize_minmax(curves)
    matrix = build_matrix(curves, method="standard")
    result = hierarchical_clustering(matrix, [c.name for c in curves], n_clusters=2)

    metrics = evaluate(curves, result["labels"])
    assert "ari" in metrics
    assert "purity" in metrics


def test_pipeline_without_labels(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    normalize_minmax(curves)
    matrix = build_matrix(curves)
    result = hierarchical_clustering(matrix, [c.name for c in curves])
    metrics = evaluate(curves, result["labels"])
    assert metrics == {}


def test_single_file_pipeline(simple_midi_file):
    curves = load_midi_files([simple_midi_file])
    normalize_minmax(curves)
    assert curves[0].points is not None

    matrix = build_matrix(curves)
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0

    result = hierarchical_clustering(matrix, [curves[0].name])
    assert result["labels"] == [0]


def test_empty_pipeline():
    curves = load_midi_files([])
    normalize_minmax(curves)
    matrix = build_matrix(curves)
    assert matrix.shape == (0, 0)


def test_non_midi_file(tmp_path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("C D E F G")
    curve = load_midi(str(bad_file))
    assert curve is None
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass across all test files

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests"
```

---

### Task 16: Final Verification

- [ ] **Step 1: Run full test suite one more time**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (should be ~25+ tests total)

- [ ] **Step 2: Verify app launches (headless check)**

Run: `python -c "from src.main import MainWindow; w = MainWindow(); print('MainWindow created OK')"`
Expected: Prints "MainWindow created OK" (no crash, may fail if no display available)

- [ ] **Step 3: Commit any final touches**

```bash
git status
git add -A
git commit -m "chore: final verification and cleanup"
```
