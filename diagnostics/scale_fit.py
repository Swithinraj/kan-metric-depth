"""Can the existing image features predict metric scale when the shape predictor and its reference are held fixed?

The whole adapters + votes model is frozen. Per image: shape = decoder log-depth minus its mean (frozen),
target log-scale = median over valid pixels of (log true depth - shape), gain fixed at zero. Small FastKAN scale
heads are trained on cached pooled backbone features with one equal-weight L1 term per image, and the scale
error |predicted - target log-scale| is reported on the training views, on unseen scenes of the training cameras
(validation), on the frozen test set and on the unseen datasets (fresh set, Middlebury). Validation picks the epoch.

Heads: residual (target minus the frozen decoder level, the current formulation), absolute (no decoder level),
absolute + level (decoder level as an input), camera only. References: training median, the frozen decoder's own
level, and the scale the current model outputs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eval_fresh import Fresh
from eval_middlebury import Middlebury
from kan_metric_depth.model import SplitScaleDepth, conditioner
from kan_metric_depth.data import MIN_DEPTH, UnifiedDepth

RUN, WORK, CACHE = "runs/split6_kan_votes", (518, 686), Path("runs/scale_cache.pt")
BAD_2021 = ("artroom", "bandsaw", "ladder", "octogons", "pendulum", "podium", "traproom")


@torch.inference_mode()
def cache(model: SplitScaleDepth, data, names: list[str], passes: int = 1, keep=None) -> dict:
    out = {k: [] for k in ("tokens", "camera", "level", "spread", "target", "current", "source")}
    device = next(model.parameters()).device
    for _ in range(passes):
        for batch in DataLoader(data, batch_size=8, num_workers=5):
            rgb, intrinsics = batch["rgb"].to(device), batch["intrinsics"].to(device)
            depth, valid = batch["depth"].to(device), batch["valid"].to(device)
            height, width = rgb.shape[-2:]
            with torch.autocast("cuda"):
                taps = model._taps(rgb)
                raw = model._decode(taps, intrinsics, (height, width)).float()
                level = raw.mean(dim=(2, 3), keepdim=True)
                shape = raw - level
                current = model._vote_parts(taps, raw, level, shape, intrinsics)[1].float().flatten()
            resid = depth.clamp_min(MIN_DEPTH).log() - shape
            fx, fy, cx, cy = intrinsics.float().unbind(dim=1)
            for i in range(rgb.shape[0]):
                if valid[i].float().mean() < 0.02:
                    continue
                out["tokens"].append(torch.cat([torch.cat((t[i, 0], t[i, 1:].mean(dim=0))) for t in taps]).float().cpu())
                out["camera"].append(torch.stack((2 * torch.log(fx[i] / width), 2 * torch.log(fy[i] / height),
                                                  2 * (cx[i] / width - 0.5), 2 * (cy[i] / height - 0.5))).cpu())
                out["level"].append(level[i].flatten().cpu())
                out["spread"].append(shape[i].std().view(1).cpu())
                out["target"].append(resid[i][valid[i]].median().view(1).cpu())
                out["current"].append(current[i].view(1).cpu())
                out["source"].append(int(batch["source"][i]))
    result = {k: torch.stack(v) for k, v in out.items() if k != "source"}
    result["source"], result["names"] = torch.tensor(out["source"]), names
    if keep is not None:
        mask = torch.tensor(keep[:len(result["source"])])
        result = {k: (v[mask] if torch.is_tensor(v) else v) for k, v in result.items()}
    return result


def build_cache() -> dict:
    device = torch.device("cuda")
    model = SplitScaleDepth("kan", WORK, lora_rank=8, votes=12).to(device).eval()
    model.load_state_dict(torch.load(Path(RUN) / "best.pt", map_location=device)["model"])
    train = UnifiedDepth("datasets", split="train", zoom_range=(0.5, 4.0), focal_range=(0.5, 2.8))
    val = UnifiedDepth("datasets", split="val", augment=False, max_per_source=80)
    test = UnifiedDepth("datasets", split="test", augment=False)
    fresh, middlebury = Fresh(), Middlebury()
    close = [not (source_id == 1 and scene.name.startswith(BAD_2021)) for source_id, scene in middlebury.items]
    sets = {}
    for name, data, passes, keep in (("val", val, 1, None), ("test", test, 1, None), ("fresh", fresh, 1, None),
                                     ("middlebury", middlebury, 1, close), ("train", train, 2, None)):
        sets[name] = cache(model, data, list(data.sources), passes, keep)
        print("cached", name, len(sets[name]["source"]), flush=True)
    torch.save(sets, CACHE)
    return sets


class Head(nn.Module):
    def __init__(self, kind: str, start: float):
        super().__init__()
        self.kind = kind
        self.content = nn.Sequential(nn.Dropout(0.1), nn.Linear(3072, 32), nn.LayerNorm(32)) if kind != "camera only" else None
        extra = {"residual": 2, "absolute + level": 2, "absolute": 0, "camera only": 0}[kind]
        self.net = conditioner("kan", (32 if self.content else 0) + extra + 4, 32, 1)
        self.start = start

    def forward(self, s: dict) -> torch.Tensor:
        parts = [s["camera"]]
        if self.content:
            parts.insert(0, self.content(s["tokens"]))
        if self.kind in ("residual", "absolute + level"):
            parts += [s["level"] - 1.0, s["spread"]]
        out = self.net(torch.cat(parts, dim=1)[:, :, None, None]).flatten(1)
        return out + (s["level"] if self.kind == "residual" else self.start)


def error(pred: torch.Tensor, s: dict, sel=None) -> float:
    diff = (pred - s["target"]).abs().flatten()
    return float(diff[sel].mean() if sel is not None else diff.mean())


def fit(kind: str, sets: dict, seed: int = 0, epochs: int = 60) -> tuple[dict, nn.Module]:
    torch.manual_seed(seed)
    train = sets["train"]
    head = Head(kind, float(train["target"].median())).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    count = len(train["source"])
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=epochs * (count // 256))
    best, state = float("inf"), None
    for _ in range(epochs):
        head.train()
        order = torch.randperm(count, device="cuda")
        for i in range(count // 256):
            idx = order[i * 256:(i + 1) * 256]
            batch = {k: v[idx] for k, v in train.items() if torch.is_tensor(v)}
            loss = (head(batch) - batch["target"]).abs().mean()   # one equal-weight term per image
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        head.eval()
        with torch.no_grad():
            val = error(head(sets["val"]), sets["val"])
        if val < best:
            best, state = val, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        return {name: head(s) for name, s in sets.items()}, head


def main() -> None:
    sets = torch.load(CACHE) if CACHE.exists() else build_cache()
    sets = {n: {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in s.items()} for n, s in sets.items()}
    order = ("train", "val", "test", "fresh", "middlebury")
    median = sets["train"]["target"].median()
    rows = {"training median (constant)": {n: median.expand_as(s["target"]) for n, s in sets.items()},
            "frozen decoder level": {n: s["level"] for n, s in sets.items()},
            "current model's scale": {n: s["current"] for n, s in sets.items()}}
    for kind in ("camera only", "absolute", "absolute + level", "residual"):
        runs = [fit(kind, sets, seed)[0] for seed in range(3)]
        rows[f"head: {kind}"] = runs
    print("\nMean |log-scale error| (0.10 is about 10% off). Heads: mean of 3 seeds.\n" + f"{'':30s}" + "".join(f"{n:>12s}" for n in order))
    for name, preds in rows.items():
        runs = preds if isinstance(preds, list) else [preds]
        print(f"{name:30s}" + "".join(f"{np.mean([error(r[n], sets[n]) for r in runs]):12.3f}" for n in order), flush=True)
    print("\nPer source, signed mean log error (negative = too close) and mean |error|:")
    for set_name in ("test", "fresh", "middlebury"):
        s = sets[set_name]
        for source_id, source in enumerate(s["names"]):
            sel = s["source"] == source_id
            if not sel.any():
                continue
            cells = []
            for name in ("current model's scale", "head: absolute", "head: residual"):
                runs = rows[name] if isinstance(rows[name], list) else [rows[name]]
                signed = np.mean([float((r[set_name] - s["target"]).flatten()[sel].mean()) for r in runs])
                cells.append(f"{name.split(': ')[-1][:14]:>14s} {signed:+.3f} / {np.mean([error(r[set_name], s, sel) for r in runs]):.3f}")
            print(f"  {set_name:10s} {source[:30]:30s} n={int(sel.sum()):4d}  " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    sys.exit(main())
