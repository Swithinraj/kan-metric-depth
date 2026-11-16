"""Metric-depth evaluation on the frozen test set: raw predictions in meters, no alignment to ground truth.

Reported per source: AbsRel, RMSE, delta1 on raw output, `bias` = median(pred / gt) (1.0 = right scale,
>1 = too far), and the same AbsRel/delta1 after a per-image median-scale fix (shape-only reference).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from kan_metric_depth.data import MIN_DEPTH, UnifiedDepth
from kan_metric_depth.metrics import KEYS, score
from kan_metric_depth.reference import predict

ROOT = Path("datasets")
OUT = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    parser.add_argument("--name", default="depth_anything_v2_metric_indoor_small")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "test_set" / "manifest.json").read_text())
    protocol = manifest["protocol"]
    test = UnifiedDepth(ROOT, split="test", augment=False, height=protocol["height"], width=protocol["width"],
                        max_depth=protocol["max_depth_m"], max_per_source=protocol["per_source_cap"])
    assert [s["rgb"] for s in manifest["samples"]] == [p["rgb"].relative_to(ROOT).as_posix() for _, p in test.items], \
        "test set no longer matches the frozen manifest"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model_id} parameters={params:,} max_depth={getattr(model.config, 'max_depth', None)}", flush=True)

    rows, start = [], time.perf_counter()
    for batch in DataLoader(test, batch_size=args.batch_size, num_workers=2):
        depth = predict(processor, model, batch["rgb"], device).cpu()
        for i in range(depth.shape[0]):
            valid, gt = batch["valid"][i, 0], batch["depth"][i, 0]
            pred = depth[i][valid].clamp(MIN_DEPTH, protocol["max_depth_m"])
            rows.append({"source": test.sources[int(batch["source"][i])], **score(pred, gt[valid])})
    seconds = time.perf_counter() - start

    sources = sorted({r["source"] for r in rows}, key=lambda s: test.sources.index(s))
    per_source = {s: {k: float(np.mean([r[k] for r in rows if r["source"] == s])) for k in KEYS} for s in sources}
    keep = [s for s in sources if s != "hypersim"]
    result = {"model": args.model_id, "parameters": params, "images": len(rows), "seconds": round(seconds, 1),
              "per_source": per_source,
              "macro": {k: float(np.mean([per_source[s][k] for s in sources])) for k in KEYS},
              "macro_without_hypersim": {k: float(np.mean([per_source[s][k] for s in keep])) for k in KEYS},
              "micro": {k: float(np.mean([r[k] for r in rows])) for k in KEYS}}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.name}_metric.json").write_text(json.dumps(result, indent=1))
    for s, m in per_source.items():
        print(f"{s:11s}", {k: round(v, 3) for k, v in m.items()})
    for name in ("macro", "macro_without_hypersim", "micro"):
        print(f"{name:23s}", {k: round(v, 3) for k, v in result[name].items()})


if __name__ == "__main__":
    main()
