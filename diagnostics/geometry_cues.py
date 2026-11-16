"""Which classical single-image scale cue would fix the hard scenes? Diagnostic on the two fresh TUM sequences.

The motion-capture pose gives each frame's gravity direction and camera height above the floor. Each model's
prediction is rescaled per image in several ways and scored (AbsRel / delta1 / bias):

raw           the prediction as it is
oracle        scale = median(gt / pred), the best any single scale can do
ground-auto   1. ground plane, no floor mask: the lowest points of the predicted cloud are assumed to be the
              floor; scale = known camera height / predicted height
ground-mask   1. ground plane with the true floor pixels: scale = median(plane depth / pred) on the floor
anchor        2. known object: one small patch (1% of the image) has known depth; scale = median(gt / pred) there
Nothing is trained; ground truth is only used where the method says so.
"""

import re
from pathlib import Path

import numpy as np
import torch

from eval_fresh import FAMILY, MAX_DEPTH, WORK, Fresh
from kan_metric_depth.model import IntrinsicsDepth, SplitScaleDepth
from kan_metric_depth.data import MIN_DEPTH

MODELS = [("Large, untouched", "none", FAMILY.format("Large"), None),
          ("Ours, adapters + votes", "split-kan-lora8-votes12", None, "runs/split6_kan_votes")]
METHODS = ("raw", "oracle", "ground-auto", "ground-mask", "anchor")
FLOOR_BAND, MIN_FLOOR = 0.06, 0.05  # metres above the floor that still count as floor; minimum floor share of the image


def rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rays(k: torch.Tensor, height: int, width: int) -> torch.Tensor:
    v, u = torch.meshgrid(torch.arange(height) + 0.5, torch.arange(width) + 0.5, indexing="ij")
    return torch.stack(((u - k[2]) / k[0], (v - k[3]) / k[1], torch.ones_like(u)), dim=-1).double()


def stats(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float, float]:
    pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
    return (float(((pred - gt).abs() / gt).mean()), float((torch.maximum(pred / gt, gt / pred) < 1.25).double().mean()),
            float((pred / gt).median()))


@torch.inference_mode()
def main() -> None:
    device, data = torch.device("cuda"), Fresh()
    frames = []  # (index, source, up-in-camera, camera position z, world heights of gt points)
    for index, (source_id, sample) in enumerate(data.items):
        if not data.sources[source_id].startswith("tum"):
            continue
        rgb_path = Path(sample[0])
        poses = np.loadtxt(rgb_path.parents[1] / "groundtruth.txt")
        row = poses[np.abs(poses[:, 0] - float(rgb_path.stem)).argmin()]
        frames.append((index, data.sources[source_id], rotation(row[4:8]), row[3]))

    # Geometry from ground truth: gravity from the pose, floor height per sequence from the lowest ground-truth points.
    geometry, lowest = {}, {}
    for index, source, rot, cam_z in frames:
        item = data[index]
        gt, valid, k = item["depth"][0].double(), item["valid"][0], item["intrinsics"]
        ray = rays(k, *gt.shape)
        up = torch.from_numpy(rot.T @ np.array([0.0, 0.0, 1.0]))            # world up, in camera coordinates
        world_z = cam_z + (ray * gt[..., None]) @ up                         # height of every gt point in the world
        geometry[index] = (item, ray, up, cam_z, world_z)
        lowest.setdefault(source, []).append(world_z[valid].quantile(0.01).item())
    floor_z = {s: float(np.percentile(v, 5)) for s, v in lowest.items()}
    print("floor height per sequence (world z):", {s: round(z, 2) for s, z in floor_z.items()}, flush=True)

    generator = torch.Generator().manual_seed(0)
    for name, cond, model_id, run in MODELS:
        if cond.startswith("split"):
            lora, votes = (int(m.group(1)) if (m := re.search(key + r"(\d+)", cond)) else 0 for key in ("lora", "votes"))
            model = SplitScaleDepth("kan", WORK, lora_rank=lora, votes=votes).to(device).eval()
            model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=device)["model"])
        else:
            model = IntrinsicsDepth(cond, WORK, model_id=model_id).to(device).eval()
        rows = []
        for index, source, _, _ in frames:
            item, ray, up, cam_z, world_z = geometry[index]
            gt, valid = item["depth"][0].double(), item["valid"][0]
            with torch.autocast("cuda"):
                pred = model(item["rgb"][None].to(device), item["intrinsics"][None].to(device)).float().exp().cpu()[0, 0].double()
            pred = pred.clamp(MIN_DEPTH, MAX_DEPTH)
            height = cam_z - floor_z[source]                                 # known camera height above the floor
            floor = valid & (world_z < floor_z[source] + FLOOR_BAND)
            share = float(floor.float().mean())
            row = {"source": source, "floor": share, "height": height, "raw": stats(pred[valid], gt[valid]),
                   "oracle": stats(pred[valid] * (gt[valid] / pred[valid]).median(), gt[valid])}
            below = -(ray * pred[..., None]) @ up                            # how far below the camera each predicted point is
            row["ground-auto"] = stats(pred[valid] * height / below.flatten().quantile(0.98).clamp_min(1e-3), gt[valid])
            if share >= MIN_FLOOR:
                plane = height / (-(ray @ up)).clamp_min(1e-3)               # depth of the floor along each ray
                row["ground-mask"] = stats(pred[valid] * (plane[floor] / pred[floor]).median(), gt[valid])
            h, w = gt.shape
            for _ in range(20):                                              # a random 10% x 10% patch with enough ground truth
                top, left = int(torch.randint(h - h // 10, (), generator=generator)), int(torch.randint(w - w // 10, (), generator=generator))
                box = torch.zeros_like(valid)
                box[top:top + h // 10, left:left + w // 10] = True
                box &= valid
                if box.sum() >= 50:
                    row["anchor"] = stats(pred[valid] * (gt[box] / pred[box]).median(), gt[valid])
                    break
            rows.append(row)
        print(name, flush=True)
        for source in sorted(floor_z) + ["ALL"]:
            for subset, sel in (("all images", [r for r in rows if source in (r["source"], "ALL")]),
                                ("floor visible", [r for r in rows if source in (r["source"], "ALL") and r["floor"] >= MIN_FLOOR])):
                if not sel:
                    continue
                cells = "  ".join(f"{m}=" + "/".join(f"{np.mean([r[m][i] for r in sel if m in r]):.3f}" for i in range(3))
                                  for m in METHODS if any(m in r for r in sel))
                print(f"    {source[:28]:28s} {subset:13s} n={len(sel):3d} floor={np.mean([r['floor'] for r in sel]):.2f} "
                      f"h={np.mean([r['height'] for r in sel]):.2f}m  {cells}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
