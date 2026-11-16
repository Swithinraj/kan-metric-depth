"""Build results/UNSEEN_COMPARISON.md from the saved per-image/per-scene scores on data no model trained on."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

MODELS = [("Small (base, untouched)", "Small, untouched", "24.8M"),
          ("Ours (Small + FastKAN)", "Split + FastKAN votes, LoRA r8, + ARKit visits, fixed sampling", "about 25M"),
          ("Large (untouched)", "Large, untouched", "335M")]
METRICS = [("AbsRel", "AbsRel (lower)"), ("RMSE", "RMSE m (lower)"), ("delta1", "delta1 (higher)"), ("bias", "bias (1 = right)"),
           ("median_AbsRel", "shape AbsRel (lower)"), ("median_delta1", "shape delta1 (higher)")]
BAD_2021 = ("artroom", "bandsaw", "ladder", "octogons", "pendulum", "podium", "traproom")
fresh = json.load(open("results/fresh_eval.json"))
mb = json.load(open("results/unseen_middlebury.json"))
lower = {"AbsRel", "RMSE", "median_AbsRel"}


def table(title, rows_of):  # rows_of(key) -> dict metric -> value
    vals = [rows_of(key) for _, key, _ in MODELS]
    out = [f"### {title}", "", "| Metric | " + " | ".join(n for n, _, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for m, label in METRICS:
        col = [v[m] for v in vals]
        if m == "bias":
            best = int(np.argmin([abs(np.log(c)) for c in col]))
        else:
            best = int(np.argmin(col) if m in lower else np.argmax(col))
        out.append(f"| {label} | " + " | ".join(f"**{c:.3f}**" if i == best else f"{c:.3f}" for i, c in enumerate(col)) + " |")
    return out + [""]


def mb_rows(select):
    def rows(key):
        scenes = [s for s in mb[key] if select(s)]
        return {m: float(np.mean([s[m] for s in scenes])) for m, _ in METRICS}
    return rows


print(Counter(s["source"] for s in mb["Large, untouched"]))
sources = sorted({s["source"] for s in mb["Large, untouched"]})
old, new = sources[0], sources[-1]
close = lambda s: s["source"] == new and not s["scene"].lower().startswith(BAD_2021)
count = lambda sel: sum(sel(s) for s in mb["Large, untouched"])
lines = ["# Unseen-data comparison: base Small, ours, Large", "",
         "None of the three models trained on any image below. Ours is the adapters + votes model trained with the "
         "extra ARKitScenes rooms (`runs/split9_kan_data`, one seed). These sets were looked at during development, "
         "so they are unseen data, not untouched data.", "",
         "Parameters: " + "; ".join(f"{n} {p}" for n, _, p in MODELS) + ".", "",
         "Scale-fixed (shape) columns rescale each prediction by its median before scoring, so they measure relative depth only.", "",
         "The two TUM sequences of the fresh set (freiburg2 desk, freiburg3 long office) are left out of this comparison; "
         "their scores remain in `results/fresh_eval.json`.", "",
         "## 7-Scenes (200 images: heads and stairs)", ""]
KEEP = [s for s in fresh["Large, untouched"]["per_source"] if not s.startswith("tum")]
lines += table("Both sequences (mean of sequences)",
               lambda k: {m: float(np.mean([fresh[k]["per_source"][s][m] for s in KEEP])) for m, _ in METRICS})
for src in KEEP:
    lines += table(src, lambda k, src=src: fresh[k]["per_source"][src])
lines += ["## Middlebury stereo (structured-light ground truth)", ""]
lines += table(f"Middlebury 2014, DSLR ({count(lambda s: s['source'] == old)} scenes)", mb_rows(lambda s: s["source"] == old))
lines += table(f"Middlebury 2021, close range ({count(close)} scenes)", mb_rows(close))
Path("results/UNSEEN_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
