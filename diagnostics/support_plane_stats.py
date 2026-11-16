"""How many images could a ground-plane loss act on, and is camera height easier to guess than scene depth?

Per image, planes are fitted to the ground-truth depth (sequential RANSAC). The support plane is the lowest plane
that lies below the camera, is roughly horizontal for an upright camera (normal within 60 degrees of image-down)
and covers at least 5% of the image. Reported per source: share of images with such a plane, the camera's height
above it, and the spread of that height against the spread of the scene's median depth (both as std of the log).
The plane fit is checked against motion-capture height on the two fresh TUM sequences. CPU only, nothing is trained.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diagnostics.geometry_cues import rays, rotation
from eval_fresh import Fresh
from kan_metric_depth.data import UnifiedDepth

MIN_SHARE, MAX_PLANES, ITERATIONS, SAMPLE = 0.05, 5, 300, 6000


def support_plane(depth: torch.Tensor, valid: torch.Tensor, k: torch.Tensor, generator: torch.Generator):
    """Returns (camera height above the plane, share of valid pixels on it, unit normal pointing from camera to plane) or None."""
    points = (rays(k, *depth.shape) * depth[..., None].double())[valid]
    total = points.shape[0]
    if total < 500:
        return None
    points = points[torch.randperm(total, generator=generator)[:SAMPLE]]
    count, best, remaining = points.shape[0], None, torch.ones(points.shape[0], dtype=torch.bool)
    for _ in range(MAX_PLANES):
        pool = points[remaining]
        if pool.shape[0] < MIN_SHARE * count:
            break
        triples = pool[torch.randint(pool.shape[0], (ITERATIONS, 3), generator=generator)]
        normals = torch.linalg.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)
        offsets = (normals * triples[:, 0]).sum(dim=1)
        tolerance = 0.015 + 0.01 * pool[:, 2]                                   # sensor noise grows with depth
        inliers = ((pool @ normals.T) - offsets).abs() < tolerance[:, None]
        top = inliers.sum(dim=0).argmax()
        mask = inliers[:, top]
        if mask.sum() < MIN_SHARE * count:
            break
        # Refit on the inliers (least squares), orient the normal away from the camera.
        centre = pool[mask].mean(dim=0)
        normal = torch.linalg.svd(pool[mask] - centre).Vh[-1]
        distance = float(normal @ centre)
        if distance < 0:
            normal, distance = -normal, -distance
        if normal[1] > 0.5 and (best is None or distance > best[0]):             # below the camera, lowest so far
            best = (distance, float(mask.sum()) / count, normal)
        index = remaining.nonzero()[:, 0][mask]
        remaining[index] = False
    return best


class Planes(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        depth, valid = item["depth"][0], item["valid"][0]
        plane = support_plane(depth, valid, item["intrinsics"], torch.Generator().manual_seed(index))
        median = float(depth[valid].median()) if valid.any() else float("nan")
        height, share, normal = plane if plane else (float("nan"), 0.0, torch.zeros(3, dtype=torch.double))
        return {"source": int(item["source"]), "median": median, "height": height, "share": share,
                "pitch": float(torch.rad2deg(torch.atan2(normal[2], normal[1]))) if plane else float("nan"), "normal": normal}


def collect(data) -> dict:
    rows = {}
    for batch in DataLoader(Planes(data), batch_size=32, num_workers=5):
        for key, value in batch.items():
            rows.setdefault(key, []).append(value.numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def report(title: str, data) -> dict:
    rows = collect(data)
    print(f"\n{title}\n{'source':28s} {'images':>6s} {'plane':>6s} {'share':>6s} {'height p10/p50/p90 (m)':>24s} {'pitch p50':>9s} "
          f"{'spread height':>13s} {'spread depth':>12s}", flush=True)
    for source_id, name in list(enumerate(data.sources)) + [(-1, "ALL")]:
        sel = (rows["source"] == source_id) | (source_id < 0)
        found = sel & (rows["share"] >= MIN_SHARE)
        if not sel.any():
            continue
        h = rows["height"][found]
        cells = (f"{np.percentile(h, 10):.2f} / {np.median(h):.2f} / {np.percentile(h, 90):.2f}" if found.sum() > 4 else "-")
        spread = f"{np.log(h).std():.2f}" if found.sum() > 4 else "-"
        pitch = f"{np.median(rows['pitch'][found]):.0f}" if found.sum() > 4 else "-"
        depth_spread = np.log(rows["median"][found]).std() if found.sum() > 4 else np.nan
        print(f"{name[:28]:28s} {sel.sum():6d} {found.mean() if source_id < 0 else found.sum() / sel.sum():6.0%} "
              f"{rows['share'][found].mean() if found.any() else 0:6.2f} {cells:>24s} {pitch:>9s} {spread:>13s} {depth_spread:12.2f}", flush=True)
    return rows


def check_tum(fresh: Fresh, rows: dict) -> None:
    """Fitted camera height against motion capture (floor level per sequence as in geometry_cues)."""
    truth, lowest = {}, {}
    for index, (source_id, sample) in enumerate(fresh.items):
        name = fresh.sources[source_id]
        if not name.startswith("tum"):
            continue
        rgb_path = Path(sample[0])
        poses = np.loadtxt(rgb_path.parents[1] / "groundtruth.txt")
        row = poses[np.abs(poses[:, 0] - float(rgb_path.stem)).argmin()]
        item = fresh[index]
        up = torch.from_numpy(rotation(row[4:8]).T @ np.array([0.0, 0.0, 1.0]))
        world_z = row[3] + (rays(item["intrinsics"], *item["depth"].shape[-2:]) * item["depth"][0].double()[..., None]) @ up
        truth[index] = (name, row[3], float((item["valid"][0] & (world_z < 1e9)).float().mean()), up, world_z, item["valid"][0])
        lowest.setdefault(name, []).append(world_z[item["valid"][0]].quantile(0.01).item())
    floor = {s: float(np.percentile(v, 5)) for s, v in lowest.items()}
    errors, hits, visible, tilt = [], 0, 0, []
    for index, (name, cam_z, _, up, world_z, valid) in truth.items():
        seen = float((valid & (world_z < floor[name] + 0.06)).float().mean()) >= MIN_SHARE
        visible += seen
        if rows["share"][index] >= MIN_SHARE:
            if seen:
                hits += 1
                errors.append(rows["height"][index] / (cam_z - floor[name]))
                tilt.append(np.degrees(np.arccos(np.clip(float(-up @ torch.from_numpy(rows["normal"][index])), -1, 1))))
    found = sum(rows["share"][i] >= MIN_SHARE for i in truth)
    print(f"\nTUM check: floor truly visible in {visible} of {len(truth)} images; a support plane was fitted in {found}; both in {hits}.")
    if errors:
        print(f"  where both: fitted / true height median {np.median(errors):.3f} (p10 {np.percentile(errors, 10):.3f}, p90 {np.percentile(errors, 90):.3f}); "
              f"normal off true gravity by median {np.median(tilt):.1f} degrees", flush=True)
    print(f"  fitted planes that are not the floor (desk or other): {found - hits}", flush=True)


def main() -> None:
    fresh = Fresh()
    check_tum(fresh, report("Fresh held-out set", fresh))
    report("Training set", UnifiedDepth("datasets", split="train", augment=False))
    report("Frozen test set", UnifiedDepth("datasets", split="test", augment=False))


if __name__ == "__main__":
    sys.exit(main())
