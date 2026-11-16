"""Freeze the held-out test set: write datasets/test_set/manifest.json and prove it shares no frame with train/val."""

import json
from pathlib import Path

from kan_metric_depth.data import UnifiedDepth

ROOT = Path("datasets")
PER_SOURCE = 100
HEIGHT, WIDTH, MAX_DEPTH = 252, 336, 20.0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if (ROOT / "test_set" / "manifest.json").exists():
        raise SystemExit("Test set is already frozen; delete datasets/test_set/manifest.json only if you mean to redefine it.")
    test = UnifiedDepth(ROOT, split="test", augment=False, height=HEIGHT, width=WIDTH, max_depth=MAX_DEPTH,
                        max_per_source=PER_SOURCE)
    frames = [(test.sources[source_id], sample) for source_id, sample in test.items]
    manifest = {
        "protocol": {"height": HEIGHT, "width": WIDTH, "max_depth_m": MAX_DEPTH, "min_depth_m": 0.1,
                     "per_source_cap": PER_SOURCE, "selection": "evenly spaced over the test pool of each source"},
        "samples": [{"source": source, "scene": s["scene"], "rgb": rel(s["rgb"]), "depth": rel(s["depth"]),
                     "K": list(s["K"]), "dist": list(s["dist"]) if s.get("dist") else None} for source, s in frames],
    }
    out = ROOT / "test_set" / "manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))

    test_ids = {rel(s["rgb"]) for _, s in frames}
    for split in ("train", "val"):
        other = UnifiedDepth(ROOT, split=split, augment=False)
        shared = test_ids & {rel(s["rgb"]) for _, s in other.items}
        scenes_t = {(src, s["scene"]) for src, s in frames if src != "nyu"}
        scenes_o = {(other.sources[i], s["scene"]) for i, s in other.items if other.sources[i] != "nyu"}
        print(f"{split}: {len(other)} samples, shared frames with test = {len(shared)}, shared scenes = {len(scenes_t & scenes_o)}")
        assert not shared and not (scenes_t & scenes_o), f"test leaks into {split}"
    print(f"test: {len(test)} samples", test.source_counts())
    scenes = {}
    for source, s in frames:
        scenes.setdefault(source, set()).add(s["scene"])
    print({k: len(v) for k, v in scenes.items()}, "scenes per source")
    print("wrote", out)


if __name__ == "__main__":
    main()
