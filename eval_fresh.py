"""Fresh held-out check: data that was never trained on and never used for any design decision.

7-Scenes (heads, stairs; Kinect, default intrinsics 585/585/320/240, depth not exactly registered to colour)
and TUM RGB-D freiburg2 desk / freiburg3 long office (other sensors than the freiburg1 one in the test set).
Raw metric depth, same 252x336 input, 518x686 working size and scoring as the frozen test set.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from kan_metric_depth.metrics import KEYS, score
from kan_metric_depth.model import IntrinsicsDepth, SplitScaleDepth
from kan_metric_depth.data import MIN_DEPTH, TUM_DIST, TUM_K, _tum_pairs, undistort, warp

ROOT = Path("datasets/fresh_eval")
OUT = Path("results/fresh_eval.json")
MAX_DEPTH, WORK, PER_SOURCE = 20.0, (518, 686), 100
FAMILY = "depth-anything/Depth-Anything-V2-Metric-Indoor-{}-hf"
MODELS = [("Small, untouched", "none", FAMILY.format("Small"), None),
          ("Large, untouched", "none", FAMILY.format("Large"), None),
          *[(f"Split + FastKAN, lens-balanced, seed {i}", "split-kan", None, f"runs/split4_kan{s}")
            for i, s in enumerate(("", "_s1", "_s2"))],
          ("Split + FastKAN, lens-balanced, LoRA r8", "split-kan-lora8", None, "runs/split5_kan_lora"),
          ("Split + FastKAN votes, lens-balanced, LoRA r8", "split-kan-lora8-votes12", None, "runs/split6_kan_votes"),
          ("Split + FastKAN votes + metre shift, LoRA r8", "split-kan-lora8-votes12-shift", None, "runs/split7_kan_shift"),
          ("Split + FastKAN votes, LoRA r16 all", "split-kan-lora16all-votes12", None, "runs/split8_kan_lora16"),
          ("Split + FastKAN votes, LoRA r8, + ARKit visits, fixed sampling", "split-kan-lora8-votes12", None, "runs/split9_kan_data"),
          ("FiLM + FastKAN (sec. 8, original data)", "kan", None, "runs/intr_kan"),
          ("Split + FastKAN, + SUN RGB-D, before tabletop data (sec. 12)", "split-kan", None, "runs/split2_kan"),
          ("Split + FastKAN, + tabletop data, before lens balance (sec. 13)", "split-kan", None, "runs/split3_kan")]


def _thin(items: list, count: int) -> list:
    return [items[i] for i in np.linspace(0, len(items) - 1, min(count, len(items))).round().astype(int)]


class Fresh(Dataset):
    def __init__(self, height: int = 252, width: int = 336):
        self.size, self.items, self.sources = (height, width), [], []
        for scene in sorted((ROOT / "seven_scenes").iterdir()):
            frames = sorted(scene.rglob("*.color.png"))
            self._add(f"7scenes_{scene.name}", [(f, f.with_name(f.name.replace("color", "depth")), (585.0, 585.0, 320.0, 240.0), None, 1000.0)
                                                 for f in frames])
        for seq in sorted((ROOT / "tum").iterdir()):
            camera = next(name for name in TUM_K if name in seq.name)
            self._add("tum_" + seq.name.split("rgbd_dataset_")[1],
                      [(rgb, depth, TUM_K[camera], TUM_DIST.get(camera), 5000.0) for rgb, depth in _tum_pairs(seq)])

    def _add(self, name: str, samples: list) -> None:
        self.sources.append(name)
        self.items.extend((len(self.sources) - 1, s) for s in _thin(samples, PER_SOURCE))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        source_id, (rgb_path, depth_path, k, dist, unit) = self.items[index]
        rgb = torch.from_numpy(np.asarray(Image.open(rgb_path).convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
        raw = np.asarray(Image.open(depth_path)).astype(np.float32)
        raw[raw >= 65535] = 0.0  # 7-Scenes marks invalid depth with 65535
        depth = torch.from_numpy(raw / unit)[None]
        if dist:
            rgb, depth = undistort(rgb, depth, k, dist)
        h, w = depth.shape[-2:]
        box_w = min(float(w), h * self.size[1] / self.size[0])
        box_h = box_w * self.size[0] / self.size[1]
        rgb, depth, k = warp(rgb, depth, k, ((w - box_w) / 2, (h - box_h) / 2, (w + box_w) / 2, (h + box_h) / 2), self.size)
        valid = torch.isfinite(depth) & (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)
        return {"rgb": rgb.clamp(0, 1), "depth": torch.where(valid, depth, torch.zeros_like(depth)), "valid": valid,
                "intrinsics": torch.tensor(k, dtype=torch.float32), "source": torch.tensor(source_id)}


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda")
    data = Fresh()
    print("images", len(data), data.sources, flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, cond, model_id, run in MODELS:
        if name in results:
            continue
        if cond.startswith("split"):
            lora, votes = (int(m.group(1)) if (m := re.search(key + r"(\d+)", cond)) else 0 for key in ("lora", "votes"))
            model = SplitScaleDepth(cond.split("-")[1], WORK, lora_rank=lora, votes=votes, metre_shift="shift" in cond, lora_all="all" in cond).to(device).eval()
        else:
            model = IntrinsicsDepth(cond, WORK, **({"model_id": model_id} if model_id else {})).to(device).eval()
        if run:
            model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device)["model"])
        rows = []
        for batch in DataLoader(data, batch_size=2, num_workers=2):
            with torch.autocast("cuda"):
                pred = model(batch["rgb"].to(device), batch["intrinsics"].to(device)).float().exp().cpu()
            for i in range(pred.shape[0]):
                valid, gt = batch["valid"][i, 0], batch["depth"][i, 0]
                if valid.float().mean() < 0.02:
                    continue
                rows.append({"source": data.sources[int(batch["source"][i])], "median_gt": float(gt[valid].median()),
                             **score(pred[i, 0][valid].clamp(MIN_DEPTH, MAX_DEPTH), gt[valid])})
        per_source = {s: {k: float(np.mean([r[k] for r in rows if r["source"] == s])) for k in KEYS} for s in data.sources}
        results[name] = {"per_source": per_source, "macro": {k: float(np.mean([v[k] for v in per_source.values()])) for k in KEYS},
                         "images": len(rows), "median_gt": {s: float(np.median([r["median_gt"] for r in rows if r["source"] == s]))
                                                           for s in data.sources}}
        m = results[name]["macro"]
        print(f"{name:42s} AbsRel={m['AbsRel']:.3f} RMSE={m['RMSE']:.3f} delta1={m['delta1']:.3f} bias={m['bias']:.3f} "
              f"shape_delta1={m['median_delta1']:.3f}", flush=True)
        for s, v in per_source.items():
            print(f"    {s:38s} AbsRel={v['AbsRel']:.3f} delta1={v['delta1']:.3f} bias={v['bias']:.3f} shape={v['median_delta1']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
