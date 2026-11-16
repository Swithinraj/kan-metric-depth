"""Score Depth Anything V2-Small on the frozen test set (datasets/test_set/manifest.json) and write a results log.

Depth Anything predicts relative disparity, so each image is aligned to ground truth before scoring:
  affine      per-image least-squares scale + shift in disparity space (the standard affine-invariant protocol)
  scale-only  per-image least-squares scale in disparity space, no shift (stricter: keeps zero disparity = infinity)
Both use ground truth, so they measure depth *shape*, not metric ability.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kan_metric_depth.reference import RELATIVE_SMALL as MODEL_ID, load_depth_anything, predict
from kan_metric_depth.data import MIN_DEPTH, UnifiedDepth

ROOT = Path("datasets")
OUT = Path("results")
KEYS = ("AbsRel", "RMSE", "delta1")


def metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    p, t = pred.double(), gt.double()
    return {"AbsRel": ((p - t).abs() / t).mean().item(), "RMSE": ((p - t) ** 2).mean().sqrt().item(),
            "delta1": (torch.maximum(p / t, t / p) < 1.25).double().mean().item()}


def align(disparity: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, max_depth: float) -> dict[str, torch.Tensor]:
    d, target = disparity[valid].double(), (1.0 / gt[valid]).double()
    scale_only = (d * target).sum() / (d * d).sum()
    a = torch.stack([d, torch.ones_like(d)], dim=-1)
    scale, shift = torch.linalg.lstsq(a, target.unsqueeze(-1)).solution.squeeze(-1)
    floor = 1.0 / max_depth
    return {"affine": 1.0 / (scale * disparity.double() + shift).clamp_min(floor),
            "scale_only": 1.0 / (scale_only * disparity.double()).clamp_min(floor)}


def main() -> None:
    manifest = json.loads((ROOT / "test_set" / "manifest.json").read_text())
    protocol = manifest["protocol"]
    test = UnifiedDepth(ROOT, split="test", augment=False, height=protocol["height"], width=protocol["width"],
                        max_depth=protocol["max_depth_m"], max_per_source=protocol["per_source_cap"])
    frames = [s["rgb"] for s in manifest["samples"]]
    assert frames == [p["rgb"].relative_to(ROOT).as_posix() for _, p in test.items], "test set no longer matches the frozen manifest"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_depth_anything(device)
    params = sum(p.numel() for p in model.parameters())
    loader = DataLoader(test, batch_size=8, num_workers=2)
    rows = []
    start = time.perf_counter()
    for batch in loader:
        disparity = predict(processor, model, batch["rgb"], device).cpu()
        for i in range(disparity.shape[0]):
            valid, gt = batch["valid"][i, 0], batch["depth"][i, 0]
            aligned = align(disparity[i], gt, valid, protocol["max_depth_m"])
            row = {"source": test.sources[int(batch["source"][i])], "median_gt_m": gt[valid].median().item(),
                   "valid_frac": valid.float().mean().item()}
            for name, pred in aligned.items():
                for key, value in metrics(pred[valid].clamp(MIN_DEPTH, protocol["max_depth_m"]), gt[valid]).items():
                    row[f"{name}/{key}"] = value
            rows.append(row)
    seconds = time.perf_counter() - start

    sources = sorted({r["source"] for r in rows}, key=lambda s: test.sources.index(s))
    table = {}
    for protocol_name in ("affine", "scale_only"):
        per_source = {s: {k: float(np.mean([r[f"{protocol_name}/{k}"] for r in rows if r["source"] == s])) for k in KEYS}
                      for s in sources}
        table[protocol_name] = {
            "per_source": per_source,
            "macro": {k: float(np.mean([per_source[s][k] for s in sources])) for k in KEYS},
            "micro": {k: float(np.mean([r[f"{protocol_name}/{k}"] for r in rows])) for k in KEYS}}
    info = {"model": MODEL_ID, "parameters": params, "images": len(rows), "seconds": round(seconds, 1),
            "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "counts": {s: sum(r["source"] == s for r in rows) for s in sources},
            "median_gt_m": {s: float(np.median([r["median_gt_m"] for r in rows if r["source"] == s])) for s in sources},
            "protocol": protocol, "results": table}
    OUT.mkdir(exist_ok=True)
    (OUT / "depth_anything_v2s_baseline.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "results"}, indent=1))
    for name, t in table.items():
        print(name, "macro", {k: round(v, 4) for k, v in t["macro"].items()}, "micro", {k: round(v, 4) for k, v in t["micro"].items()})
        for s, m in t["per_source"].items():
            print("   ", s, {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
