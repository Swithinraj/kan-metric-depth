"""One loader for every collected indoor RGB-D source, each sample carrying its camera intrinsics.

Returns single frames: rgb [3,H,W] in 0..1, planar metric depth [1,H,W] in
meters (0 = invalid), valid [1,H,W] bool, intrinsics [4] = (fx, fy, cx, cy) in
output pixels, and a source id. Geometry augmentation (zoom in/out, crop, flip)
rewrites the intrinsics to match, which is what lets a model see many virtual
lenses from a handful of real cameras. Colour jitter never touches geometry.
"""

import csv
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

SOURCES = ("bonn", "nyu", "arkit", "diode", "hypersim", "tum_indoor", "sun_kv2", "sun_rs", "sun_xtion", "bop_long", "bop_std", "arkit_train")
EVAL_ONLY = {"tum_indoor"}
MIN_DEPTH = 0.1

# (fx, fy, cx, cy) at the native resolution of the RGB image each source ships.
BONN_K = (542.822841, 542.576870, 315.593520, 237.756098)
NYU_K = (518.857901, 519.469611, 325.582449, 253.736166)
DIODE_K = (886.81, 927.06, 512.0, 384.0)
TUM_K = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
         "freiburg3": (535.4, 539.2, 320.1, 247.6)}
# OpenCV order (k1, k2, p1, p2, k3). Bonn and TUM freiburg1/2 ship distorted RGB (depth is registered to it);
# freiburg3 and NYU labeled frames are already undistorted. Sources: TUM file_formats page, IPB Bonn page.
BONN_DIST = (0.039903, -0.099343, -0.000730, -0.000144, 0.0)
TUM_DIST = {"freiburg1": (0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
            "freiburg2": (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)}

ROOTS = {
    "bonn": "bonn_rgbd_dynamic/extracted/rgbd_bonn_dataset",
    "tum_indoor": "tum_indoor/extracted",
    "nyu": "nyu_v2_val/nyu_test",
    "arkit": "arkitscenes_sub/Validation",
    "arkit_train": "arkitscenes_sub/Training",  # more visits from the Training fold; never part of the frozen test set
    "diode": "diode_val/extracted/val/indoors",
    "hypersim": "hypersim_sub",
    # SUN RGB-D without its Kinect v1 part (that part contains NYU, including frames of the frozen test set).
    "sun_kv2": "sunrgbd/kv2",
    "sun_rs": "sunrgbd/realsense",
    "sun_xtion": "sunrgbd/xtion",
    # Close-range tabletop captures from the BOP benchmark sets (real sensors, per-image intrinsics).
    "bop_long": "bop",  # long lenses: YCB-Video, T-LESS
    "bop_std": "bop",   # ordinary lenses: Linemod-Occluded, HOPE, HomebrewedDB, TUD-L
}
BOP_SETS = {"bop_long": {"ycbv": 1, "tless": 1}, "bop_std": {"lmo": 4, "hope": 1, "hb": 1, "tudl": 2}}  # name -> keep every n-th
DEFAULT_STRIDE = {"bonn": 10, "tum_indoor": 5, "sun_kv2": 3, "sun_xtion": 3}


def _tum_pairs(seq_dir: Path, tolerance: float = 0.02) -> list[tuple[Path, Path]]:
    def read(name: str) -> list[tuple[float, str]]:
        rows = []
        for line in (seq_dir / name).read_text().splitlines():
            if line and not line.startswith("#"):
                stamp, rel = line.split(maxsplit=1)
                rows.append((float(stamp), rel))
        return [(t, r) for t, r in rows if (seq_dir / r).exists()]

    rgb, depth = read("rgb.txt"), read("depth.txt")
    stamps = np.array([t for t, _ in rgb])
    used, pairs = set(), []
    for stamp, depth_rel in depth:
        i = int(np.searchsorted(stamps, stamp))
        j = min((k for k in (i - 1, i) if 0 <= k < len(stamps)), key=lambda k: abs(stamps[k] - stamp), default=None)
        if j is not None and j not in used and abs(stamps[j] - stamp) < tolerance:
            used.add(j)
            pairs.append((seq_dir / rgb[j][1], seq_dir / depth_rel))
    return pairs


def _tum_samples(source: str, root: Path) -> list[dict]:
    samples = []
    for seq_dir in sorted(p for p in root.iterdir() if (p / "rgb.txt").exists()):
        if source == "bonn":
            intrinsics, dist = BONN_K, BONN_DIST
        else:
            camera = next(name for name in TUM_K if name in seq_dir.name)
            intrinsics, dist = TUM_K[camera], TUM_DIST.get(camera)
        for rgb, depth in _tum_pairs(seq_dir):
            samples.append({"scene": seq_dir.name, "rgb": rgb, "depth": depth, "K": intrinsics, "dist": dist})
    return samples


def _nyu_samples(root: Path) -> list[dict]:
    depths = {p.stem.rsplit("_", 1)[-1]: p for p in (root / "depth").glob("*.npy")}
    return [{"scene": "nyu", "rgb": p, "depth": depths[p.stem.rsplit("_", 1)[-1]], "K": NYU_K}
            for p in sorted((root / "image").glob("*.png")) if p.stem.rsplit("_", 1)[-1] in depths]


def _arkit_samples(root: Path) -> list[dict]:
    samples = []
    for video in sorted(p for p in root.iterdir() if p.is_dir()):
        pincams = {float(p.stem.split("_", 1)[1]): p for p in video.rglob("*.pincam")}
        stamps = np.array(sorted(pincams))
        for rgb in sorted((video / "wide").glob("*.png")):
            depth = video / "highres_depth" / rgb.name
            if not depth.exists() or not len(stamps):
                continue
            stamp = float(rgb.stem.split("_", 1)[1])
            nearest = float(stamps[int(np.abs(stamps - stamp).argmin())])
            w, _, fx, fy, cx, cy = (float(v) for v in pincams[nearest].read_text().split())
            with Image.open(rgb) as image:
                scale = image.size[0] / w
            samples.append({"scene": video.name, "rgb": rgb, "depth": depth,
                            "K": (fx * scale, fy * scale, cx * scale, cy * scale)})
    return samples


def _diode_samples(root: Path) -> list[dict]:
    samples = []
    for rgb in sorted(root.rglob("*.png")):
        depth = rgb.with_name(rgb.stem + "_depth.npy")
        if depth.exists():
            samples.append({"scene": f"{rgb.parent.parent.name}/{rgb.parent.name}", "rgb": rgb, "depth": depth,
                            "mask": rgb.with_name(rgb.stem + "_depth_mask.npy"), "K": DIODE_K})
    return samples


def _hypersim_cameras(root: Path) -> dict[str, tuple[float, float, float, float]]:
    """Per-scene pinhole K from Apple's projection matrices; tilt-shift scenes (|M20|, |M21| > 0.005) are left out."""
    cameras = {}
    with open(root / "metadata_camera_parameters.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["M_proj_20"])) > 0.005 or abs(float(row["M_proj_21"])) > 0.005:
                continue
            w, h = float(row["settings_output_img_width"]), float(row["settings_output_img_height"])
            cameras[row["scene_name"]] = (float(row["M_proj_00"]) * w / 2, float(row["M_proj_11"]) * h / 2, w / 2, h / 2)
    return cameras


def _hypersim_samples(root: Path) -> list[dict]:
    samples = []
    cameras = _hypersim_cameras(root)
    for scene in sorted(p for p in root.iterdir() if p.is_dir() and p.name in cameras):
        images = scene / "images"
        for rgb in sorted((images / "scene_cam_00_final_preview").glob("frame.*.color.jpg")):
            frame = rgb.name.split(".")[1]
            depth = images / "scene_cam_00_geometry_hdf5" / f"frame.{frame}.depth_meters.hdf5"
            if depth.exists():
                samples.append({"scene": scene.name, "rgb": rgb, "depth": depth, "K": cameras[scene.name]})
    return samples


def _sun_samples(root: Path) -> list[dict]:
    """SUN RGB-D: one folder per image with image/, depth_bfx/ and a per-image intrinsics.txt (3x3, row-major).

    Scene = the SUN3D sequence folder where there is one; elsewhere captures are only time-ordered, so
    runs of 20 consecutive captures are kept together.
    """
    samples = []
    folders = sorted(p.parent for p in root.rglob("intrinsics.txt"))
    for index, folder in enumerate(folders):
        k = [float(v) for v in (folder / "intrinsics.txt").read_text().split()]
        group = folder.parent.relative_to(root).as_posix()
        scene = group if "sun3ddata" in group else f"{group}/{index // 20:04d}"
        samples.append({"scene": scene, "rgb": next((folder / "image").iterdir()), "depth": next((folder / "depth_bfx").iterdir()),
                        "K": (k[0], k[4], k[2], k[5])})
    return samples


def _bop_samples(source: str, root: Path) -> list[dict]:
    """BOP layout: <set>/<scene>/{rgb,depth}/<id>.png and a scene_camera.json with cam_K and depth_scale (to mm)."""
    samples = []
    for name, step in BOP_SETS[source].items():
        for camera_file in sorted((root / name).rglob("scene_camera.json")):
            cameras = json.loads(camera_file.read_text())
            scene_dir = root / name / camera_file.parent.name
            for index, frame in enumerate(sorted(cameras, key=int)[::step]):
                k = cameras[frame]["cam_K"]
                rgb = next(scene_dir.joinpath("rgb").glob(f"{int(frame):06d}.*"))
                # Linemod-Occluded is one long video; cut it into blocks so validation gets a share.
                scene = f"{name}/{camera_file.parent.name}" + (f"/{index // 50}" if name == "lmo" else "")
                samples.append({"scene": scene, "rgb": rgb, "depth": scene_dir / "depth" / f"{int(frame):06d}.png",
                                "K": (k[0], k[4], k[2], k[5]), "depth_unit": cameras[frame]["depth_scale"] / 1000.0})
    return samples


def _planar_from_euclidean(distance: np.ndarray, k: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = k
    height, width = distance.shape
    u = np.arange(width, dtype=np.float32)[None, :] + 0.5
    v = np.arange(height, dtype=np.float32)[:, None] + 0.5
    return distance / np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)


def read_sample(source: str, sample: dict) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    with Image.open(sample["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    k = sample["K"]
    if source in ("bonn", "tum_indoor"):
        depth = np.asarray(Image.open(sample["depth"]), dtype=np.float32) / 5000.0
    elif source.startswith("arkit"):
        depth = np.asarray(Image.open(sample["depth"]), dtype=np.float32) / 1000.0
    elif source.startswith("bop_"):
        depth = np.asarray(Image.open(sample["depth"]), dtype=np.float32) * sample["depth_unit"]
    elif source.startswith("sun_"):
        raw = np.asarray(Image.open(sample["depth"])).astype(np.uint16)
        depth = ((raw >> 3) | (raw << 13)).astype(np.float32) / 1000.0  # SUN RGB-D stores millimeters rotated by 3 bits
    elif source == "hypersim":
        with h5py.File(sample["depth"], "r") as h5:
            depth = np.nan_to_num(np.asarray(h5["dataset"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth = _planar_from_euclidean(depth, k)
    else:
        depth = np.load(sample["depth"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if "mask" in sample:
            depth = np.where(np.load(sample["mask"]) > 0, depth, 0.0).astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB {rgb.shape[:2]} and depth {depth.shape} differ for {sample['rgb']}")
    return rgb, depth, k


def undistortion_grid(height: int, width: int, k: tuple[float, float, float, float],
                      dist: tuple[float, float, float, float, float]) -> torch.Tensor:
    """Sampling grid that maps each ideal (undistorted) pixel to where it sits in the distorted image (OpenCV model)."""
    fx, fy, cx, cy = k
    k1, k2, p1, p2, k3 = dist
    u = torch.arange(width, dtype=torch.float64)[None, :].expand(height, width)
    v = torch.arange(height, dtype=torch.float64)[:, None].expand(height, width)
    x, y = (u - cx) / fx, (v - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    src_u, src_v = fx * xd + cx, fy * yd + cy
    return torch.stack(((src_u + 0.5) / width * 2 - 1, (src_v + 0.5) / height * 2 - 1), dim=-1)[None].float()


def undistort(rgb: torch.Tensor, depth: torch.Tensor, k: tuple[float, float, float, float],
              dist: tuple[float, float, float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove lens distortion, keeping the intrinsics k. Depth (registered to the RGB) is resampled nearest-neighbour."""
    grid = undistortion_grid(rgb.shape[-2], rgb.shape[-1], k, dist)
    return (F.grid_sample(rgb[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0],
            F.grid_sample(depth[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0])


def warp(rgb: torch.Tensor, depth: torch.Tensor, k: tuple[float, float, float, float], box: tuple[float, float, float, float],
         out_size: tuple[int, int], flip: bool = False) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float, float]]:
    """Resample the source box (x0, y0, x1, y1), possibly extending outside the image, to out_size.

    Planar depth values are unchanged by crop/scale/flip; only the intrinsics move.
    Out-of-image area is zero RGB and invalid (0) depth.
    """
    out_h, out_w = out_size
    height, width = rgb.shape[-2:]
    x0, y0, x1, y1 = box
    fx, fy, cx, cy = k
    # Pre-shrink with area averaging so a 1920-wide frame isn't point-sampled down to ~340 px.
    shrink = min(1.0, 2.0 * out_w / (x1 - x0))
    if shrink < 1.0:
        new_h, new_w = max(1, round(height * shrink)), max(1, round(width * shrink))
        sx, sy = new_w / width, new_h / height
        rgb = F.interpolate(rgb[None], size=(new_h, new_w), mode="area")[0]
        depth = F.interpolate(depth[None], size=(new_h, new_w), mode="nearest")[0]
        x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        height, width = new_h, new_w
    a, c = (x1 - x0) / width, (y1 - y0) / height
    theta = torch.tensor([[-a if flip else a, 0.0, (x0 + x1) / width - 1.0],
                          [0.0, c, (y0 + y1) / height - 1.0]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, out_h, out_w), align_corners=False)
    rgb_out = F.grid_sample(rgb[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    depth_out = F.grid_sample(depth[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    scale_x, scale_y = out_w / (x1 - x0), out_h / (y1 - y0)
    new_cx, new_cy = (cx - x0) * scale_x, (cy - y0) * scale_y
    if flip:
        new_cx = out_w - new_cx
    return rgb_out, depth_out, (fx * scale_x, fy * scale_y, new_cx, new_cy)


class UnifiedDepth(Dataset):
    """Mix of every collected source. `split` is 'train', 'val' or 'test'.

    Scenes (NYU: frames) are partitioned deterministically and never shared: sorted scene i with
    i % 4 == 3 is test, i % 4 == 1 is val, the rest train (NYU: frame % 8 == 7 test, == 3 val).
    Once <root>/test_set/manifest.json exists, the test split is exactly its frames and its scenes
    stay out of train/val, whatever data is added later. EVAL_ONLY sources are test only. `max_per_source` thins a split to evenly spaced
    frames. `zoom_range` > 1 crops in
    (longer virtual focal length), < 1 zooms out with invalid padding (shorter).
    """

    def __init__(self, root: str | Path = "datasets", sources: tuple[str, ...] = SOURCES, split: str = "train",
                 height: int = 252, width: int = 336, augment: bool | None = None, max_depth: float = 20.0,
                 stride: dict[str, int] | None = None, zoom_range: tuple[float, float] = (0.7, 2.0),
                 min_valid_fraction: float = 0.02, max_per_source: int | None = None,
                 focal_range: tuple[float, float] | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.root, self.split = Path(root), split
        self.height, self.width, self.max_depth = height, width, max_depth
        self.augment = (split == "train") if augment is None else augment
        self.zoom_range, self.min_valid_fraction = zoom_range, min_valid_fraction
        # focal_range: draw the *output* focal length per image width from this range for every image, whatever
        # camera it came from, so lens and scene content are independent in training (zoom is clipped to zoom_range).
        self.focal_range = focal_range
        stride = {**DEFAULT_STRIDE, **(stride or {})}
        self.sources = [s for s in sources if not (split == "train" and s in EVAL_ONLY)]
        self.items: list[tuple[int, dict]] = []
        manifest = self.root / "test_set" / "manifest.json"
        frozen = None
        if manifest.exists():
            entries = json.loads(manifest.read_text())["samples"]
            frozen = {"scenes": {(e["source"], e["scene"]) for e in entries}, "frames": [e["rgb"] for e in entries]}
        for source_id, source in enumerate(self.sources):
            path = self.root / ROOTS[source]
            if not path.is_dir():
                raise FileNotFoundError(f"{source}: {path} not found")
            if source in ("bonn", "tum_indoor"):
                samples = _tum_samples(source, path)
            elif source.startswith("sun_"):
                samples = _sun_samples(path)
            elif source.startswith("bop_"):
                samples = _bop_samples(source, path)
            else:
                samples = {"nyu": _nyu_samples, "arkit": _arkit_samples, "arkit_train": _arkit_samples, "diode": _diode_samples,
                           "hypersim": _hypersim_samples}[source](path)
            scenes = sorted({s["scene"] for s in samples})
            if frozen is None:
                scene_part = {scene: {3: "test", 1: "val"}.get(i % 4, "train") for i, scene in enumerate(scenes)}
            else:
                # Test scenes are whatever the manifest froze; data added later can only land in train/val.
                test_scenes = {scene for src, scene in frozen["scenes"] if src == source}
                rest = [scene for scene in scenes if scene not in test_scenes]
                scene_part = {scene: "val" if i % 3 == 1 else "train" for i, scene in enumerate(rest)}
                scene_part.update({scene: "test" for scene in test_scenes})
            if frozen is not None and split == "test":
                by_path = {s["rgb"].relative_to(self.root).as_posix(): s for s in samples}
                self.items.extend((source_id, by_path[p]) for p in frozen["frames"] if p in by_path)
                continue
            chosen = []
            for index, sample in enumerate(samples):
                if source in EVAL_ONLY:
                    part = "test"
                elif source == "nyu":
                    part = {7: "test", 3: "val"}.get(index % 8, "train")
                else:
                    part = scene_part[sample["scene"]]
                if part == split and index % stride.get(source, 1) == 0:
                    chosen.append(sample)
            if max_per_source and len(chosen) > max_per_source:
                picks = np.linspace(0, len(chosen) - 1, max_per_source).round().astype(int)
                chosen = [chosen[i] for i in picks]
            self.items.extend((source_id, sample) for sample in chosen)
        if not self.items:
            raise ValueError("No samples found")

    def __len__(self) -> int:
        return len(self.items)

    def source_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in self.sources}
        for source_id, _ in self.items:
            counts[self.sources[source_id]] += 1
        return counts

    def balanced_sampler(self, num_samples: int | None = None, power: float = 0.0) -> WeightedRandomSampler:
        """A source with n frames is drawn with probability proportional to n ** power.

        power=0: equal probability per source (a tiny source is then repeated very often); power=1: every frame equal.
        """
        per_source = self.source_counts()
        weights = [per_source[self.sources[source_id]] ** (power - 1.0) for source_id, _ in self.items]
        return WeightedRandomSampler(weights, num_samples or len(self.items), replacement=True)

    def _box(self, width: int, height: int, fx: float = 0.0) -> tuple[tuple[float, float, float, float], bool]:
        base_w = min(float(width), height * self.width / self.height)
        base_h = base_w * self.height / self.width
        zoom = 1.0
        if self.augment:
            low, high = self.zoom_range
            if self.focal_range:
                target = np.exp(random.uniform(np.log(self.focal_range[0]), np.log(self.focal_range[1])))
                zoom = float(np.clip(target / (fx / base_w), low, high))
            else:
                zoom = float(np.exp(random.uniform(np.log(low), np.log(high))))
        box_w, box_h = base_w / zoom, base_h / zoom
        if self.augment:
            x0 = random.uniform(0.0, width - box_w) if box_w <= width else random.uniform(width - box_w, 0.0)
            y0 = random.uniform(0.0, height - box_h) if box_h <= height else random.uniform(height - box_h, 0.0)
        else:
            x0, y0 = (width - box_w) / 2, (height - box_h) / 2
        return (x0, y0, x0 + box_w, y0 + box_h), self.augment and random.random() < 0.5

    @staticmethod
    def _jitter(rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb * (1.0 + random.uniform(-0.3, 0.3))
        mean = rgb.mean()
        rgb = (rgb - mean) * (1.0 + random.uniform(-0.3, 0.3)) + mean
        luma = rgb.mean(dim=0, keepdim=True)
        rgb = luma + (rgb - luma) * random.uniform(0.6, 1.3)
        return rgb

    def _load(self, index: int) -> dict[str, torch.Tensor] | None:
        source_id, sample = self.items[index]
        rgb, depth, k = read_sample(self.sources[source_id], sample)
        height, width = depth.shape
        box, flip = self._box(width, height, k[0])
        rgb_t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth)[None]
        if sample.get("dist"):
            rgb_t, depth_t = undistort(rgb_t, depth_t, k, sample["dist"])
        rgb_t, depth_t, k = warp(rgb_t, depth_t, k, box, (self.height, self.width), flip)
        if self.augment:
            rgb_t = self._jitter(rgb_t)
        valid = torch.isfinite(depth_t) & (depth_t >= MIN_DEPTH) & (depth_t <= self.max_depth)
        if valid.float().mean() < self.min_valid_fraction:
            return None
        return {"rgb": rgb_t.clamp(0.0, 1.0), "depth": torch.where(valid, depth_t, torch.zeros_like(depth_t)),
                "valid": valid, "intrinsics": torch.tensor(k, dtype=torch.float32),
                "source": torch.tensor(source_id)}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        for offset in range(len(self.items)):
            out = self._load((index + offset) % len(self.items))
            if out is not None:
                return out
        raise RuntimeError("No sample with enough valid depth")
