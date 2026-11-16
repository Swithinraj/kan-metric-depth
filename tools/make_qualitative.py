"""Example predictions of the untouched Small, our model and the untouched Large on images none of them trained on.

python -m tools.make_qualitative predict    run the three models, cache to runs/qualitative.npz (needs torch + GPU)
python -m tools.make_qualitative plot       draw docs/figures/qualitative.png from the cache (needs matplotlib)

The images are fixed picks made before looking at the predictions: the middle frame of each 7-Scenes sequence,
two Middlebury scenes and the first ARKitScenes / DIODE image of the frozen test set.
"""

import sys
from pathlib import Path

import numpy as np

CACHE = Path("runs/qualitative.npz")
OUT = Path("docs/figures/qualitative.png")
MODELS = ("Small (untouched)", "Ours", "Large (untouched)")
RUN = "runs/split9_kan_data"


def predict() -> None:
    import torch

    from eval_fresh import FAMILY, MAX_DEPTH, WORK, Fresh
    from eval_middlebury import Middlebury
    from kan_metric_depth.data import MIN_DEPTH, UnifiedDepth
    from kan_metric_depth.metrics import score
    from kan_metric_depth.model import IntrinsicsDepth, SplitScaleDepth

    fresh, middlebury = Fresh(), Middlebury()
    test = UnifiedDepth("datasets", split="test", augment=False)
    picks = [("7-Scenes heads", fresh[50]), ("7-Scenes stairs", fresh[150])]
    for scene in ("Piano", "chess2"):
        index = next(i for i, (_, d) in enumerate(middlebury.items) if d.name == scene)
        picks.append((f"Middlebury {scene}", middlebury[index]))
    for source in ("arkit", "diode"):
        index = next(i for i, (s, _) in enumerate(test.items) if test.sources[s] == source)
        picks.append((f"Test set, {'ARKitScenes' if source == 'arkit' else 'DIODE'}", test[index]))

    device = torch.device("cuda")
    out = {"names": np.array([n for n, _ in picks]),
           "rgb": np.stack([s["rgb"].permute(1, 2, 0).numpy() for _, s in picks]),
           "gt": np.stack([s["depth"][0].numpy() for _, s in picks]),
           "valid": np.stack([s["valid"][0].numpy() for _, s in picks])}
    preds, errors = [], []
    for kind in ("Small", "ours", "Large"):
        if kind == "ours":
            model = SplitScaleDepth("kan", WORK, lora_rank=8, votes=12)
            model.load_state_dict(torch.load(Path(RUN) / "best.pt", map_location="cpu")["model"])
        else:
            model = IntrinsicsDepth("none", WORK, model_id=FAMILY.format(kind))
        model = model.to(device).eval()
        print(kind, f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
        row, err = [], []
        for _, s in picks:
            with torch.inference_mode(), torch.autocast("cuda"):
                depth = model(s["rgb"][None].to(device), s["intrinsics"][None].to(device)).float().exp()[0, 0].cpu()
            depth = depth.clamp(MIN_DEPTH, MAX_DEPTH)
            row.append(depth.numpy())
            err.append(score(depth[s["valid"][0]], s["depth"][0][s["valid"][0]])["AbsRel"])
        preds.append(np.stack(row))
        errors.append(err)
        del model
        torch.cuda.empty_cache()
    out["pred"], out["absrel"] = np.stack(preds), np.array(errors)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(out["names"], out["absrel"].round(3), sep="\n")


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.make_figures import INK, MUTED, SURFACE

    d = np.load(CACHE)
    rows = len(d["names"])
    fig, axes = plt.subplots(rows, 5, figsize=(13.2, 2.1 * rows + 0.4), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.17})
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#e9e8e4")
    heads = ["Input", "Ground truth", *MODELS]
    for r in range(rows):
        gt = np.where(d["valid"][r], d["gt"][r], np.nan)
        pooled = np.concatenate([gt[d["valid"][r]]] + [d["pred"][m, r].ravel() for m in range(3)])
        low, high = np.percentile(pooled, 1), np.percentile(pooled, 99)   # one colour scale for the whole row
        best = int(np.argmin(d["absrel"][:, r]))
        panels = [d["rgb"][r], gt] + [d["pred"][m, r] for m in range(3)]
        notes = [str(d["names"][r]), f"{np.nanpercentile(gt, 2):.1f} to {np.nanpercentile(gt, 98):.1f} m"] +                 [f"AbsRel {d['absrel'][m, r]:.2f}" + ("  (lowest)" if m == best else "") for m in range(3)]
        for c, ax in enumerate(axes[r]):
            ax.imshow(panels[c]) if c == 0 else ax.imshow(panels[c], cmap=cmap, vmin=low, vmax=high)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(notes[c], fontsize=9, color=INK, pad=3, loc="left",
                         fontweight="bold" if c >= 2 and c - 2 == best else "normal")
            if r == 0:
                ax.text(0, 1.16, heads[c], transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    predict() if sys.argv[1:] == ["predict"] else plot()
