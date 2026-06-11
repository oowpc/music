"""Download GTZAN dataset (1000 tracks, 10 genres) for melody classification.

Tries multiple sources in order, stops at first success:
  1. HuggingFace direct raw file (main branch)
  2. HuggingFace huggingface_hub API
  3. Kaggle via kagglehub
  4. Torchaudio built-in downloader

Expected output: data/audio/gtzan/{blues,classical,country,...,rock}/*.wav
"""
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

GTZAN_DIR = Path("data/audio/gtzan")
GTZAN_DIR.mkdir(parents=True, exist_ok=True)


def _extract_and_verify(tgz_path: Path) -> bool:
    """Extract tar.gz and verify genre subdirectories exist."""
    with tarfile.open(tgz_path) as tar:
        tar.extractall(GTZAN_DIR)
    tgz_path.unlink()
    genre_dirs = [d.name for d in sorted(GTZAN_DIR.iterdir()) if d.is_dir()]
    if genre_dirs:
        print(f"  ✅ Extracted. Genres: {genre_dirs}")
        return True
    return False


# ---- Method 1: HuggingFace raw URL ----
print("[1] Trying HuggingFace raw URL...")
try:
    url = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/genres.tar.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status == 200:
            tgz = GTZAN_DIR / "genres.tar.gz"
            with open(tgz, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"  Downloaded: {tgz.stat().st_size / 1024 / 1024:.0f} MB")
            if _extract_and_verify(tgz):
                exit(0)
        else:
            print(f"  HTTP {resp.status}")
except Exception as e:
    print(f"  {type(e).__name__}")

# ---- Method 2: huggingface_hub API ----
print("[2] Trying huggingface_hub API...")
try:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files("marsyas/gtzan")
    archives = [f for f in files if f.endswith((".tar.gz", ".tar", ".zip"))]
    print(f"  Archives found: {archives or 'none'}")

    for arch in archives:
        try:
            local_path = hf_hub_download(
                "marsyas/gtzan", arch, repo_type="dataset",
                local_dir=str(GTZAN_DIR),
            )
            print(f"  Downloaded: {local_path}")
            tgz = Path(GTZAN_DIR) / arch
            if _extract_and_verify(tgz):
                exit(0)
        except Exception as e2:
            print(f"  {arch}: {type(e2).__name__}")

    if not archives:
        # Try Parquet extraction via datasets library
        from datasets import load_dataset
        ds = load_dataset("marsyas/gtzan", split="train", trust_remote_code=True)
        print(f"  Loaded {len(ds)} tracks from Parquet")
        audio_col = [k for k, v in ds.features.items() if hasattr(v, 'sampling_rate')] or ["audio"]
        label_col = [k for k in ds.features.items() if isinstance(k[1], (str, type(None)))]
        # ... complex extraction skipped for now, fall through to next method
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# ---- Method 3: Kaggle via kagglehub ----
print("[3] Trying Kaggle...")
try:
    import kagglehub
    path = kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification"
    )
    print(f"  Downloaded to: {path}")
    src = Path(path)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        dest = GTZAN_DIR / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print("  ✅ Copied to data/audio/gtzan/")
    exit(0)
except ImportError:
    print("  kagglehub not installed (pip install kagglehub)")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# ---- Method 4: torchaudio ----
print("[4] Trying torchaudio...")
try:
    import torchaudio
    from torchaudio.datasets import GTZAN as GTZAN_Torch
    ds = GTZAN_Torch(root=str(GTZAN_DIR), download=True)
    print(f"  ✅ Loaded {len(ds)} tracks")
    exit(0)
except ImportError:
    print("  torchaudio not installed")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# ---- Fallback ----
print("\n❌ All automatic methods failed.")
print("   Manual download:")
print("   1. Kaggle: andradaolteanu/gtzan-dataset-music-genre-classification")
print("   2. HuggingFace: marsyas/gtzan")
print(f"   3. Extract to: {GTZAN_DIR.resolve()}")
print(f"   4. Expected: data/audio/gtzan/{{blues,classical,...,rock}}/*.wav")
