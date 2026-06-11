import logging
import os
import urllib.request
import zipfile

from music21 import converter, midi as m21_midi, note, stream, tempo

from src.io.midi_loader import load_midi
from src.models.melody_curve import MelodyCurve
from src.models.note import Note

logger = logging.getLogger(__name__)

ESSEN_URL = (
    "https://github.com/ccarh/"
    "essen-folksong-collection/archive/refs/heads/main.zip"
)

COLORS = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8",
    "#00cec9", "#fab1a0", "#81ecec", "#55efc4", "#74b9ff",
    "#e17055", "#6c5ce7", "#00b894", "#e84393", "#0984e3",
    "#fdcb6e", "#636e72", "#d63031", "#2d3436", "#b2bec3",
]

_SYNTHETIC_TUNES: dict[str, list[int]] = {
    "deutsch": [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60],
    "chinese": [60, 62, 64, 67, 69, 71, 72, 71, 69, 67, 64, 62, 60],
    "asmat": [60, 59, 57, 55, 53, 55, 57, 59, 60, 62, 64, 62, 60],
}


def _write_midi(filepath: str, pitches: list[int], bpm: float = 120.0) -> None:
    """Write a simple monophonic MIDI file with one note per beat."""
    score = stream.Stream()
    score.append(tempo.MetronomeMark(number=bpm))
    part = stream.Part()
    for index, pitch in enumerate(pitches):
        midi_note = note.Note(pitch)
        midi_note.duration.quarterLength = 1.0
        midi_note.volume.velocity = 80
        part.insert(float(index), midi_note)
    score.insert(0, part)
    midi_file = m21_midi.translate.streamToMidiFile(score)
    midi_file.open(filepath, "wb")
    midi_file.write()
    midi_file.close()


def _generate_synthetic_essen(target_dir: str) -> None:
    """Generate a small synthetic Essen-like dataset with region labels."""
    os.makedirs(target_dir, exist_ok=True)
    for region, pitches in _SYNTHETIC_TUNES.items():
        region_dir = os.path.join(target_dir, region)
        os.makedirs(region_dir, exist_ok=True)
        filepath = os.path.join(region_dir, f"{region}_1.mid")
        _write_midi(filepath, pitches)
    logger.info("Generated synthetic Essen dataset in %s", target_dir)


def _count_data_files(root_dir: str) -> int:
    """Count .mid/.midi or .krn files anywhere under root_dir."""
    count = 0
    for _, _, files in os.walk(root_dir):
        for filename in files:
            lower = filename.lower()
            if lower.endswith((".mid", ".midi", ".krn")):
                count += 1
    return count


def download_essen(target_dir: str = "data/standard/essen") -> str:
    """Download the Essen collection, falling back to synthetic data.

    Returns the path to the directory containing the data.  If MIDI files
    already exist under *target_dir* the download is skipped.  When the
    remote source is unreachable or contains no usable MIDI files a small
    synthetic subset is generated instead.
    """
    os.makedirs(target_dir, exist_ok=True)

    if _count_data_files(target_dir) > 0:
        logger.info("Essen data files already present in %s", target_dir)
        return target_dir

    try:
        logger.info("Attempting to download Essen collection from %s", ESSEN_URL)
        request = urllib.request.Request(ESSEN_URL, headers={"User-Agent": "music-essen-loader/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with open(os.path.join(target_dir, "essen.zip"), "wb") as zip_file:
                zip_file.write(response.read())

        with zipfile.ZipFile(os.path.join(target_dir, "essen.zip"), "r") as archive:
            archive.extractall(target_dir)
        os.remove(os.path.join(target_dir, "essen.zip"))

        if _count_data_files(target_dir) > 0:
            logger.info("Successfully downloaded and extracted Essen collection")
            return target_dir
    except Exception:
        logger.warning(
            "Download of Essen collection failed; generating synthetic dataset",
            exc_info=True,
        )

    _generate_synthetic_essen(target_dir)
    return target_dir


def _parse_krn_to_curve(filepath: str) -> MelodyCurve | None:
    """Parse a Humdrum **kern file into a MelodyCurve.

    Extracts notes (onset, pitch, velocity=80 default) from the music21
    stream.  Returns *None* when the file has no usable notes.
    """
    try:
        score = converter.parse(filepath)
    except Exception:
        logger.debug("Failed to parse %s", filepath, exc_info=True)
        return None

    raw_notes: list[Note] = []
    for element in score.flatten().notes:
        onset = float(element.getOffsetInHierarchy(score))
        pitch = element.pitch.midi if element.pitch else 60
        velocity = getattr(element.volume, "velocity", 80) or 80
        raw_notes.append(Note(timestamp=onset, pitch=pitch, velocity=velocity))

    if not raw_notes:
        return None

    name = os.path.splitext(os.path.basename(filepath))[0]
    return MelodyCurve(
        name=name,
        filepath=filepath,
        raw_notes=raw_notes,
    )


def _resolve_label(filepath: str, root_dir: str) -> str:
    """Derive a region label from the file's path relative to *root_dir*.

    The Essen repo groups files by continent → country → subregion.  This
    function picks the country-level directory as the label.  For example::

        europa/deutschl/fink/fink001.krn  →  deutschl
        europa/elsass/elsass01.krn        →  elsass
        asia/china/han/han01.krn          →  china
        america/usa/usa01.krn             →  usa
        africa/arabic01.krn               →  africa
    """
    rel = os.path.relpath(filepath, root_dir)
    parts = rel.replace(os.sep, "/").split("/")
    if parts and parts[0].startswith("essen-folksong-collection"):
        parts = parts[1:]
    if len(parts) >= 3:
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return os.path.basename(os.path.dirname(filepath))


def load_essen_collection(
    root_dir: str,
    max_files: int | None = None,
    shuffle: bool = False,
) -> list[MelodyCurve]:
    """Walk *root_dir* for .mid/.midi/.krn files and load each as a MelodyCurve.

    The parent directory name of each data file is used as the curve label
    (representing the region/category).  Colors are assigned cyclically from
    ``COLORS`` by unique label so that every curve sharing the same label
    receives the same colour.

    Set *max_files* to limit the number of loaded curves (useful for quick
    smoke-tests).  When *shuffle* is True and *max_files* is set, file paths
    are collected first, then randomly sampled — this avoids label skew
    caused by alphabetical walk order.
    """
    curves: list[MelodyCurve] = []
    label_color_map: dict[str, str] = {}
    color_index = 0

    if shuffle and max_files is not None:
        import random
        filepaths: list[tuple[str, str]] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                lower = filename.lower()
                if lower.endswith((".mid", ".midi", ".krn")):
                    filepaths.append((dirpath, filename))
        random.shuffle(filepaths)
        for dirpath, filename in filepaths:
            if len(curves) >= max_files:
                return curves
            filepath = os.path.join(dirpath, filename)
            if filename.lower().endswith(".krn"):
                curve = _parse_krn_to_curve(filepath)
            else:
                curve = load_midi(filepath)
            if curve is None:
                continue
            label = _resolve_label(filepath, root_dir)
            curve.label = label
            if label not in label_color_map:
                label_color_map[label] = COLORS[color_index % len(COLORS)]
                color_index += 1
            curve.color = label_color_map[label]
            curves.append(curve)
        return curves

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            lower = filename.lower()
            if not lower.endswith((".mid", ".midi", ".krn")):
                continue
            if max_files is not None and len(curves) >= max_files:
                return curves

            filepath = os.path.join(dirpath, filename)

            if lower.endswith(".krn"):
                curve = _parse_krn_to_curve(filepath)
            else:
                curve = load_midi(filepath)

            if curve is None:
                continue

            label = _resolve_label(filepath, root_dir)
            curve.label = label

            if label not in label_color_map:
                label_color_map[label] = COLORS[color_index % len(COLORS)]
                color_index += 1
            curve.color = label_color_map[label]

            curves.append(curve)

    return curves


def get_essen_summary(curves: list[MelodyCurve]) -> dict[str, int]:
    """Return ``{label: count}`` for every unique label in *curves*."""
    summary: dict[str, int] = {}
    for curve in curves:
        label = curve.label or "unlabeled"
        summary[label] = summary.get(label, 0) + 1
    return summary
