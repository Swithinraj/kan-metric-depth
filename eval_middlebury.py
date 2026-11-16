"""Unseen-dataset, unseen-camera check on Middlebury stereo 2014 + 2021 (evaluation only, never trained or tuned on).

Raw metric depth, same scoring and 252x336 / 518x686 protocol as the frozen test set. Ground truth is
structured-light disparity converted with Z = baseline * f / (disparity + doffs).
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
from kan_metric_depth.data import MIN_DEPTH, warp

ROOT = Path("datasets/middlebury")
MAX_DEPTH = 20.0
WORK = (518, 686)
FAMILY = "depth-anything/Depth-Anything-V2-Metric-Indoor-{}-hf"
MODELS = [("Small, untouched", "none", FAMILY.format("Small"), None),
          ("Base, untouched", "none", FAMILY.format("Base"), None),
          ("Large, untouched", "none", FAMILY.format("Large"), None),
          ("Small + fine-tune, no intrinsics", "none", None, "runs/intr_none"),
          *[(f"Small + MLP, seed {i}", "mlp", None, f"runs/intr_mlp{s}") for i, s in enumerate(("", "_s1", "_s2"))],
          *[(f"Small + FastKAN, seed {i}", "kan", None, f"runs/intr_kan{s}") for i, s in enumerate(("", "_s1", "_s2"))],
          ("Split + FastKAN", "split-kan", None, "runs/split_kan"), ("Split + MLP", "split-mlp", None, "runs/split_mlp"),
          ("Split + FastKAN, no camera", "split-kan-nocam", None, "runs/split_kan_nocam"),
          ("Split + FastKAN, more data + aug", "split-kan", None, "runs/split2_kan"),
          ("Split + MLP, more data + aug", "split-mlp", None, "runs/split2_mlp"),
          ("Split + FastKAN, + close-range data", "split-kan", None, "runs/split3_kan"),
          ("Split + FastKAN, lens-balanced", "split-kan", None, "runs/split4_kan"),
          ("Split + FastKAN, lens-balanced, seed 1", "split-kan", None, "runs/split4_kan_s1"),
          ("Split + FastKAN, lens-balanced, seed 2", "split-kan", None, "runs/split4_kan_s2"),
          ("Split + FastKAN, lens-balanced, LoRA r8", "split-kan-lora8", None, "runs/split5_kan_lora"),
          ("Split + FastKAN votes, lens-balanced, LoRA r8", "split-kan-lora8-votes12", None, "runs/split6_kan_votes"),
          ("Split + FastKAN votes + metre shift, LoRA r8", "split-kan-lora8-votes12-shift", None, "runs/split7_kan_shift"),
          ("Split + FastKAN votes, LoRA r16 all", "split-kan-lora16all-votes12", None, "runs/split8_kan_lora16"),
          ("Split + FastKAN votes, LoRA r8, + ARKit visits, fixed sampling", "split-kan-lora8-votes12", None, "runs/split9_kan_data")]
OUT = Path("results/unseen_middlebury.json")


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        assert f.readline().strip() == b"Pf"
        width, height = map(int, f.readline().split())
        endian = "<" if float(f.readline()) < 0 else ">"
        return np.frombuffer(f.read(), endian + "f4").reshape(height, width)[::-1].astype(np.float32)


class Middlebury(Dataset):
    sources = ["middlebury2014", "middlebury2021"]

    def __init__(self, height: int = 252, width: int = 336):
        self.size = (height, width)
        self.items = [(i, d) for i, year in enumerate(("2014", "2021")) for d in sorted((ROOT / year).iterdir())]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        source_id, scene = self.items[index]
        calib = dict(line.split("=", 1) for line in (scene / "calib.txt").read_text().splitlines() if "=" in line)
        f, _, cx, _, _, cy = (float(v) for v in re.findall(r"[-\d.]+", calib["cam0"])[:6])
        disparity = read_pfm(scene / "disp0.pfm")
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = float(calib["baseline"]) * f / (disparity + float(calib["doffs"])) / 1000.0
        depth[~np.isfinite(disparity) | (disparity <= 0)] = 0.0
        rgb = torch.from_numpy(np.asarray(Image.open(scene / "im0.png").convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
        h, w = depth.shape
        box_w = min(float(w), h * self.size[1] / self.size[0])
        box_h = box_w * self.size[0] / self.size[1]
        box = ((w - box_w) / 2, (h - box_h) / 2, (w + box_w) / 2, (h + box_h) / 2)
        rgb, depth_t, k = warp(rgb, torch.from_numpy(depth)[None], (f, f, cx, cy), box, self.size)
        valid = torch.isfinite(depth_t) & (depth_t >= MIN_DEPTH) & (depth_t <= MAX_DEPTH)
        return {"rgb": rgb.clamp(0, 1), "depth": torch.where(valid, depth_t, torch.zeros_like(depth_t)), "valid": valid,
                "intrinsics": torch.tensor(k, dtype=torch.float32), "source": torch.tensor(source_id), "index": index}


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda")
    data = Middlebury()
    print("images", len(data), flush=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, cond, model_id, run in MODELS:
        if name in results:  # already scored
            continue
        if cond.startswith("split"):
            lora, votes = (int(m.group(1)) if (m := re.search(key + r"(\d+)", cond)) else 0 for key in ("lora", "votes"))
            model = SplitScaleDepth(cond.split("-")[1], WORK, use_camera=not cond.endswith("nocam"), lora_rank=lora, votes=votes, metre_shift="shift" in cond, lora_all="all" in cond).to(device).eval()
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
                rows.append({"scene": data.items[int(batch["index"][i])][1].name, "source": data.sources[int(batch["source"][i])],
                             "focal_per_width": float(batch["intrinsics"][i, 0]) / data.size[1],
                             "median_gt": float(gt[valid].median()),
                             **score(pred[i, 0][valid].clamp(MIN_DEPTH, MAX_DEPTH), gt[valid])})
        results[name] = rows
        mean = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
        print(f"{name:34s} AbsRel={mean['AbsRel']:.3f} RMSE={mean['RMSE']:.3f} delta1={mean['delta1']:.3f} "
              f"bias={mean['bias']:.3f} shape_delta1={mean['median_delta1']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
