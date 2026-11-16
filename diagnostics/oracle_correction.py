"""How much could a per-image correction buy? Oracle fits against ground truth, per image, on the fresh set.

scale:      gt ~ s * pred                      (what the scale branch can fix today)
log-affine: log gt ~ a * log pred + b          (scale + gain, what the branch outputs today)
metre:      gt ~ s * pred + t                  (scale + shift in metres, relative-error weighted)
Reports AbsRel / delta1 after each oracle fit and the fitted shift t. Diagnostic only, nothing is trained.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_fresh import FAMILY, MAX_DEPTH, WORK, Fresh
from kan_metric_depth.model import IntrinsicsDepth, SplitScaleDepth
from kan_metric_depth.data import MIN_DEPTH

MODELS = [("Small, untouched", "none", FAMILY.format("Small"), None),
          ("Large, untouched", "none", FAMILY.format("Large"), None),
          ("Ours, adapters", "split-kan-lora8", None, "runs/split5_kan_lora"),
          ("Ours, adapters + votes", "split-kan-lora8-votes12", None, "runs/split6_kan_votes"),
          ("Ours, adapters + votes + metre shift", "split-kan-lora8-votes12-shift", None, "runs/split7_kan_shift")]


def fits(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred, gt = pred.double(), gt.double()
    out = {"raw": pred, "scale": pred * (gt / pred).median()}
    lp, lg = pred.log(), gt.log()
    a = ((lp - lp.mean()) * (lg - lg.mean())).sum() / ((lp - lp.mean()) ** 2).sum().clamp_min(1e-9)
    out["log-affine"] = (a * (lp - lp.mean()) + lg.mean()).exp()
    w = 1.0 / gt ** 2  # weighted least squares = relative error
    design = torch.stack((pred, torch.ones_like(pred)), dim=1)
    s, t = torch.linalg.solve((design * w[:, None]).T @ design, (design * w[:, None]).T @ gt)
    out["metre"] = (s * pred + t).clamp_min(MIN_DEPTH)
    stats = {}
    for name, p in out.items():
        ratio = torch.maximum(p / gt, gt / p)
        stats[name] = (float(((p - gt).abs() / gt).mean()), float((ratio < 1.25).double().mean()))
    stats["t"], stats["s"] = float(t), float(s)
    return stats


@torch.inference_mode()
def main() -> None:
    device, data = torch.device("cuda"), Fresh()
    for name, cond, model_id, run in MODELS:
        if run and not (Path(run) / "best.pt").exists():
            continue
        if cond.startswith("split"):
            lora, votes = (int(m.group(1)) if (m := re.search(key + r"(\d+)", cond)) else 0 for key in ("lora", "votes"))
            model = SplitScaleDepth("kan", WORK, lora_rank=lora, votes=votes, metre_shift="shift" in cond).to(device).eval()
        else:
            model = IntrinsicsDepth(cond, WORK, model_id=model_id).to(device).eval()
        if run:
            model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device)["model"])
        rows = []
        for batch in DataLoader(data, batch_size=2, num_workers=2):
            with torch.autocast("cuda"):
                pred = model(batch["rgb"].to(device), batch["intrinsics"].to(device)).float().exp().cpu()
            for i in range(pred.shape[0]):
                valid, gt = batch["valid"][i, 0], batch["depth"][i, 0]
                if valid.float().mean() >= 0.02:
                    rows.append((data.sources[int(batch["source"][i])], fits(pred[i, 0][valid].clamp(MIN_DEPTH, MAX_DEPTH), gt[valid])))
        print(name, flush=True)
        for source in data.sources + ["ALL"]:
            sel = [r for s, r in rows if source in (s, "ALL")]
            cells = "  ".join(f"{k}={np.mean([r[k][0] for r in sel]):.3f}/{np.mean([r[k][1] for r in sel]):.3f}"
                              for k in ("raw", "scale", "log-affine", "metre"))
            print(f"    {source:38s} {cells}  t={np.median([r['t'] for r in sel]):+.2f} m  s={np.median([r['s'] for r in sel]):.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
