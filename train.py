"""Train / evaluate the intrinsics-conditioned metric depth model. Run `python train.py --help`.

Validation picks the checkpoint; the frozen test set is only scored with --test (after training).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kan_metric_depth.metrics import KEYS, score
from kan_metric_depth.model import IntrinsicsDepth, SplitScaleDepth
from kan_metric_depth.data import MIN_DEPTH, UnifiedDepth

MAX_DEPTH = 20.0


def metric_loss(log_pred: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, edge_weight: float = 0.5,
                center: bool = False, per_image: bool = False) -> torch.Tensor:
    if per_image:  # every image counts the same, however many valid pixels it has
        terms = [metric_loss(log_pred[i:i + 1], depth[i:i + 1], valid[i:i + 1], edge_weight, center)
                 for i in range(log_pred.shape[0]) if valid[i].any()]
        return torch.stack(terms).mean() if terms else log_pred.sum() * 0.0
    """Scale-aware L1 on log-depth (meters, no alignment) + multi-scale gradient matching on log-depth.

    center=True removes each image's median error first, which makes the loss blind to scale (shape only).
    """
    log_gt = torch.where(valid, depth.clamp_min(MIN_DEPTH).log(), torch.zeros_like(depth))
    diff = log_pred - log_gt
    if center:
        diff = diff - torch.stack([d[v].median() if v.any() else d.new_zeros(()) for d, v in zip(diff, valid)]).view(-1, 1, 1, 1)
    diff = torch.where(valid, diff, torch.zeros_like(diff))
    loss = diff.abs().sum() / valid.sum().clamp_min(1)
    mask = valid.float()
    edge = 0.0
    for _ in range(4):
        dx, mx = diff[..., :, 1:] - diff[..., :, :-1], mask[..., :, 1:] * mask[..., :, :-1]
        dy, my = diff[..., 1:, :] - diff[..., :-1, :], mask[..., 1:, :] * mask[..., :-1, :]
        edge = edge + ((dx.abs() * mx).sum() + (dy.abs() * my).sum()) / (mx.sum() + my.sum()).clamp_min(1)
        diff, mask = diff[..., ::2, ::2], mask[..., ::2, ::2]
    return loss + edge_weight * edge


@torch.inference_mode()
def evaluate(model: IntrinsicsDepth, dataset: UnifiedDepth, device: torch.device, focal_factor: float = 1.0,
             batch_size: int = 8) -> dict:
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=3):
        intrinsics = batch["intrinsics"].to(device).clone()
        intrinsics[:, :2] *= focal_factor
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(batch["rgb"].to(device), intrinsics).float().exp().cpu()
        for i in range(pred.shape[0]):
            valid, gt = batch["valid"][i, 0], batch["depth"][i, 0]
            rows.append({"source": dataset.sources[int(batch["source"][i])],
                         **score(pred[i, 0][valid].clamp(MIN_DEPTH, MAX_DEPTH), gt[valid])})
    sources = [s for s in dataset.sources if any(r["source"] == s for r in rows)]
    per_source = {s: {k: float(np.mean([r[k] for r in rows if r["source"] == s])) for k in KEYS} for s in sources}
    real = [s for s in sources if s != "hypersim"]
    return {"per_source": per_source, "images": len(rows),
            "macro_without_hypersim": {k: float(np.mean([per_source[s][k] for s in real])) for k in KEYS}}


def show(tag: str, result: dict) -> None:
    m = result["macro_without_hypersim"]
    print(f"{tag} macro(no hypersim) AbsRel={m['AbsRel']:.4f} RMSE={m['RMSE']:.4f} delta1={m['delta1']:.4f} bias={m['bias']:.3f} "
          f"| scale-fixed AbsRel={m['median_AbsRel']:.4f} delta1={m['median_delta1']:.4f}", flush=True)
    for s, v in result["per_source"].items():
        print(f"    {s:11s} AbsRel={v['AbsRel']:.3f} RMSE={v['RMSE']:.3f} delta1={v['delta1']:.3f} bias={v['bias']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=["none", "mlp", "kan"], default="kan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5, help="pretrained neck + head")
    parser.add_argument("--cond-lr", type=float, default=1e-3, help="new conditioning modules")
    parser.add_argument("--work-height", type=int, default=392)
    parser.add_argument("--work-width", type=int, default=518)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--val-per-source", type=int, default=80)
    parser.add_argument("--model-id", default=None, help="override the pretrained checkpoint (reference runs)")
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--test", action="store_true", help="skip training; score best.pt on the frozen test set")
    parser.add_argument("--init-only", action="store_true", help="skip training; score the untrained model on test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=["film", "split"], default="film",
                        help="film: camera-only FiLM conditioning; split: shape decoder + camera-and-content scale branch")
    parser.add_argument("--zoom-min", type=float, default=0.7)
    parser.add_argument("--zoom-max", type=float, default=2.0)
    parser.add_argument("--focal-min", type=float, default=None,
                        help="with --focal-max: draw every training image's output focal length per width from this range")
    parser.add_argument("--focal-max", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=0, help="split only: low-rank adapters on the backbone attention")
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--source-power", type=float, default=0.0,
                        help="a source with n frames is drawn with probability ~ n ** power (0: equal per source)")
    parser.add_argument("--per-image-loss", action="store_true", help="average the losses per image, not per valid pixel of the batch")
    parser.add_argument("--lora-all", action="store_true", help="adapters on query, key, value and output (default: query and value)")
    parser.add_argument("--votes", type=int, default=0, help="split only: per-patch votes over this many log-scale hypotheses")
    parser.add_argument("--vote-weight", type=float, default=0.2)
    parser.add_argument("--metre-shift", action="store_true", help="with --votes: also predict a per-image shift in metres")
    parser.add_argument("--work-jitter", action="store_true",
                        help="train at a random working size per step (392x518 up to the given one), intrinsics unchanged")
    parser.add_argument("--no-camera", action="store_true", help="split only: hide the intrinsics from the scale branch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    extra = {"model_id": args.model_id} if args.model_id else {}
    if args.arch == "split":
        model = SplitScaleDepth(args.cond, (args.work_height, args.work_width), use_camera=not args.no_camera, lora_rank=args.lora_rank, votes=args.votes, metre_shift=args.metre_shift, lora_all=args.lora_all, **extra).to(device)
    else:
        model = IntrinsicsDepth(args.cond, (args.work_height, args.work_width), **extra).to(device)
    pretrained, new = model.trainable_groups()
    lora = model.lora_parameters() if args.lora_rank else []
    print(f"cond={args.cond} work={args.work_height}x{args.work_width} trainable pretrained={sum(p.numel() for p in pretrained):,} "
          f"new={sum(p.numel() for p in new):,} lora={sum(p.numel() for p in lora):,}", flush=True)

    if args.test or args.init_only:
        if args.test:
            model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
        test = UnifiedDepth("datasets", split="test", augment=False)
        result = {"test": evaluate(model, test, device, batch_size=args.eval_batch)}
        show("TEST", result["test"])
        if args.cond != "none" and not args.no_camera:
            for factor in (0.75, 1.5):
                result[f"test_focal_x{factor}"] = evaluate(model, test, device, focal_factor=factor)
                bias = result[f"test_focal_x{factor}"]["macro_without_hypersim"]["bias"] / result["test"]["macro_without_hypersim"]["bias"]
                print(f"intrinsics-use check: fx,fy x{factor} -> predicted depth x{bias:.3f}", flush=True)
        (out / ("init_test.json" if args.init_only else "test.json")).write_text(json.dumps(result, indent=1))
        return

    train = UnifiedDepth("datasets", split="train", zoom_range=(args.zoom_min, args.zoom_max),
                         focal_range=(args.focal_min, args.focal_max) if args.focal_min and args.focal_max else None)
    full_work = (args.work_height, args.work_width)
    work_sizes = [(14 * h, 14 * w) for h, w in ((28, 37), (31, 41), (34, 45)) if 14 * h < full_work[0]] + [full_work]
    val = UnifiedDepth("datasets", split="val", augment=False, max_per_source=args.val_per_source)
    print(f"train={len(train)} {train.source_counts()} val={len(val)} {val.source_counts()}", flush=True)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=train.balanced_sampler(args.samples_per_epoch, args.source_power),
                        num_workers=args.num_workers, drop_last=True, persistent_workers=True)
    groups = [{"params": pretrained, "lr": args.lr}]
    if new:
        groups.append({"params": new, "lr": args.cond_lr})
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = args.epochs * len(loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, start = 0.0, time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            rgb, depth = batch["rgb"].to(device), batch["depth"].to(device)
            valid, intrinsics = batch["valid"].to(device), batch["intrinsics"].to(device)
            if args.work_jitter:
                model.work_size = work_sizes[int(torch.randint(len(work_sizes), ()))]
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                if args.arch == "split":
                    shape, scale, gain = model.parts(rgb, intrinsics)
                else:
                    log_pred = model(rgb, intrinsics)
            if args.arch == "split":
                # Shape loss (scale-blind) trains the decoder; the metric loss sees a detached shape, so it trains the branch only.
                loss = (metric_loss(shape, depth, valid, center=True, per_image=args.per_image_loss)
                        + metric_loss(model.compose(shape.detach(), scale, gain), depth, valid, edge_weight=0.0,
                                      per_image=args.per_image_loss))
                if args.votes:
                    loss = loss + args.vote_weight * model.vote_loss(shape, gain, depth, valid)
            else:
                loss = metric_loss(log_pred.float(), depth, valid)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pretrained + new + lora, 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += loss.item()
            if step % 50 == 0 or step == len(loader):
                print(f"epoch={epoch} step={step}/{len(loader)} loss={running / step:.4f} "
                      f"it/s={step / (time.perf_counter() - start):.2f}", flush=True)
        model.work_size = full_work
        result = evaluate(model, val, device)
        show(f"VAL epoch={epoch}", result)
        history.append({"epoch": epoch, "train_loss": running / len(loader), **result["macro_without_hypersim"]})
        (out / "history.json").write_text(json.dumps(history, indent=1))
        payload = {"model": model.state_dict(), "cond": args.cond, "arch": args.arch, "use_camera": not args.no_camera, "lora_rank": args.lora_rank, "votes": args.votes, "lora_all": args.lora_all, "metre_shift": args.metre_shift, "work_size": (args.work_height, args.work_width), "epoch": epoch}
        torch.save(payload, out / "last.pt")
        if result["macro_without_hypersim"]["AbsRel"] < best:
            best = result["macro_without_hypersim"]["AbsRel"]
            torch.save(payload, out / "best.pt")
    print(f"best val AbsRel={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
