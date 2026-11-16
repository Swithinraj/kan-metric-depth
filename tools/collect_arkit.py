"""Download a visit-diverse subset of ARKitScenes upsampling (Apple LiDAR depth) plus per-frame intrinsics.

python -m tools.collect_arkit COUNT [FOLD]    FOLD is Validation (default) or Training; one video per visit.
"""
import csv, io, shutil, sys, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
OUT = Path("datasets/arkitscenes_sub")
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
FOLD = sys.argv[2] if len(sys.argv) > 2 else "Validation"
MIN_FREE_GB = 170  # stop well above the 150 GB that must stay free

rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(f"{BASE}/upsampling/metadata.csv").read().decode())))
seen, picks = set(), []
for r in rows:
    if r["fold"] == FOLD and r["visit_id"] not in seen:
        seen.add(r["visit_id"])
        picks.append(r["video_id"])
print("visits in fold:", len(picks), flush=True)
picks = picks[:COUNT]
print("videos:", len(picks), flush=True)


def fetch(vid: str) -> None:
    dest = OUT / FOLD / vid
    try:
        if shutil.disk_usage(OUT).free / 2 ** 30 < MIN_FREE_GB:
            print("SKIPPED (disk floor)", vid, flush=True)
            return
        if not (dest / "wide").exists():
            zpath = OUT / "raw" / f"{vid}.zip"
            zpath.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{BASE}/upsampling/{FOLD}/{vid}.zip", zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(OUT / FOLD)
            zpath.unlink()
        if not (dest / "vga_wide_intrinsics").exists():
            data = urllib.request.urlopen(f"{BASE}/raw/{FOLD}/{vid}/vga_wide_intrinsics.zip").read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest)
        print("done", vid, flush=True)
    except Exception as error:
        print("FAILED", vid, error, flush=True)


with ThreadPoolExecutor(6) as pool:
    list(pool.map(fetch, picks))
