"""Draw the README charts from the saved scores in results/.  python -m tools.make_figures  (needs matplotlib)"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS, OUT = Path("results"), Path("docs/figures")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, LARGE, SMALL = "#2a78d6", "#eb6834", "#1baf7a"   # validated together (colour-blind safe), fixed per model
MODELS = [("Small, untouched (24.8M)", SMALL), ("Ours (25.0M)", OURS), ("Large, untouched (335M)", LARGE)]
BAD_2021 = ("artroom", "bandsaw", "ladder", "octogons", "pendulum", "podium", "traproom")
OUR_KEY = "Split + FastKAN votes, LoRA r8, + ARKit visits, fixed sampling"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def run(name: str) -> dict:
    return load(f"runs/{name}_test.json")["test"]


fresh, middlebury = load("fresh_eval.json"), load("unseen_middlebury.json")
TEST = [run("init_518_init"), run("split9_kan_data"), run("ref_Large_518_init")]
KEYS3 = ["Small, untouched", OUR_KEY, "Large, untouched"]


def seven_scenes(key: str, metric: str) -> float:
    return float(np.mean([v[metric] for s, v in fresh[key]["per_source"].items() if s.startswith("7scenes")]))


def mb(key: str, metric: str, close: bool) -> float:
    scenes = [s for s in middlebury[key] if (s["source"] == "middlebury2021" and not s["scene"].lower().startswith(BAD_2021))
              == close and (close or s["source"] == "middlebury2014")]
    return float(np.mean([s[metric] for s in scenes]))


def benchmark_values(metric: str) -> dict[str, list[float]]:
    return {"Frozen test set\n(432 images)": [t["macro_without_hypersim"][metric] for t in TEST],
            "7-Scenes\n(200 images)": [seven_scenes(k, metric) for k in KEYS3],
            "Middlebury 2014\n(23 scenes)": [mb(k, metric, False) for k in KEYS3],
            "Middlebury 2021\nclose range (11)": [mb(k, metric, True) for k in KEYS3]}


def clean(ax, grid: str = "y") -> None:
    for side in ("top", "right", "left" if grid == "y" else "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped(ax, values: dict[str, list[float]], better: str, digits: int = 2) -> None:
    width = 0.25
    for g, (_, vals) in enumerate(values.items()):
        best = int(np.argmin(vals) if better == "lower" else np.argmax(vals))
        for m, v in enumerate(vals):
            x = g + (m - 1) * (width + 0.02)
            ax.bar(x, v, width, color=MODELS[m][1], linewidth=0)
            ax.text(x, v, f"{v:.{digits}f}", ha="center", va="bottom", fontsize=8.5, color=INK,
                    fontweight="bold" if m == best else "normal")
    ax.set_xticks(range(len(values)), list(values), fontsize=9, color=INK)
    ax.set_ylim(0, max(max(v) for v in values.values()) * 1.13)
    clean(ax)


def legend(fig, y: float = 0.955) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    fig.legend(handles, [n for n, _ in MODELS], loc="upper left", bbox_to_anchor=(0.06, y), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.1, handleheight=0.9, columnspacing=2.0)


def headline() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.2), gridspec_kw={"hspace": 0.5})
    for ax, (metric, title, better, digits) in zip(axes, (("AbsRel", "Absolute relative error (AbsRel), lower is better", "lower", 3),
                                                           ("RMSE", "RMSE in metres, lower is better", "lower", 2),
                                                           ("delta1", "Share of pixels within 25% of the truth (δ1), higher is better", "higher", 3))):
        grouped(ax, benchmark_values(metric), better, digits)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    fig.suptitle("Raw metric depth on four benchmarks, no alignment to ground truth", x=0.065, y=0.985, ha="left",
                 fontsize=13, fontweight="bold")
    legend(fig)
    fig.subplots_adjust(top=0.895, left=0.065, right=0.985, bottom=0.05)
    fig.savefig(OUT / "headline.png", dpi=160)


def parameters() -> None:
    refs = [("Small", run("init_518_init")), ("Base", run("ref_Base_518_init")), ("Large", run("ref_Large_518_init"))]
    sizes = [24.785, 97.471, 335.316]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"wspace": 0.22})
    panels = (("AbsRel on the frozen test set (lower is better)", [r["macro_without_hypersim"]["AbsRel"] for _, r in refs],
               TEST[1]["macro_without_hypersim"]["AbsRel"]),
              ("AbsRel on Middlebury 2014, unseen cameras",
               [mb(k, "AbsRel", False) for k in ("Small, untouched", "Base, untouched", "Large, untouched")], mb(OUR_KEY, "AbsRel", False)))
    for ax, (title, ref, ours) in zip(axes, panels):
        ax.plot(sizes, ref, color=LARGE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label="Depth Anything V2 metric indoor, untouched")
        ax.plot([25.027], [ours], color=OURS, marker="o", markersize=11, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", label="Ours (Small + FastKAN scale)")
        for (name, _), x, y in zip(refs, sizes, ref):
            ax.annotate(f"{name}  {y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
        ax.annotate(f"Ours  {ours:.3f}", (25.027, ours), textcoords="offset points", xytext=(12, -4), fontsize=9.5,
                    fontweight="bold", color=INK)
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 335], ["25M", "50M", "100M", "200M", "335M"])
        ax.set_xlim(17, 480)
        ax.set_ylim(0, max(ref) * 1.25)
        ax.set_xlabel("Parameters (log scale)")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=10)
        clean(ax)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.9, left=0.055, right=0.985, bottom=0.13)
    fig.savefig(OUT / "parameters.png", dpi=160)


def scale_bias() -> None:
    values = benchmark_values("bias")
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.axvline(1.0, color=INK, linewidth=1.2)
    for row, (name, vals) in enumerate(values.items()):
        ax.plot([min(vals + [1.0]), max(vals + [1.0])], [row, row], color=GRID, linewidth=2, zorder=1)
        for m, v in enumerate(vals):
            ax.plot(v, row, "o", color=MODELS[m][1], markersize=11, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f"{(v - 1) * 100:+.0f}%", (v, row), textcoords="offset points", xytext=(0, -17 if m == 2 else 10),
                        ha="center", fontsize=8.5, color=INK, fontweight="bold" if m == 1 else "normal")
    ax.set_yticks(range(len(values)), [n.replace("\n", " ") for n in values], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(values) - 0.45, -0.6)
    ax.set_xscale("log")
    ticks = [0.6, 0.8, 1.0, 1.25, 1.5]
    ax.set_xticks(ticks, ["40% too close", "20% too close", "correct", "25% too far", "50% too far"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.55, 1.75)
    clean(ax, "x")
    ax.set_title("Scale bias: median of predicted / true depth", loc="left", fontsize=12, fontweight="bold", pad=38)
    handles = [plt.Line2D([], [], marker="o", color=c, linestyle="none", markersize=9) for _, c in MODELS]
    fig.legend(handles, [n for n, _ in MODELS], loc="upper left", bbox_to_anchor=(0.225, 0.875), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(top=0.76, left=0.235, right=0.975, bottom=0.1)
    fig.savefig(OUT / "scale_bias.png", dpi=160)


def progress() -> None:
    stages = [("Small,\nuntouched", "init_518_init", "Small, untouched"),
              ("+ FastKAN on\nintrinsics", "intr_kan", "Small + FastKAN, seed 0"),
              ("split shape\n/ scale", "split_kan", "Split + FastKAN"),
              ("+ more\ncameras", "split2_kan", "Split + FastKAN, more data + aug"),
              ("+ close-range\ndata", "split3_kan", "Split + FastKAN, + close-range data"),
              ("lens-balanced\nsampling", "split4_kan", "Split + FastKAN, lens-balanced"),
              ("+ backbone\nadapters", "split5_kan_lora", "Split + FastKAN, lens-balanced, LoRA r8"),
              ("+ per-patch\nscale votes", "split6_kan_votes", "Split + FastKAN votes, lens-balanced, LoRA r8"),
              ("+ more rooms,\nfixed sampling", "split9_kan_data", OUR_KEY)]
    test = [run(r)["macro_without_hypersim"]["AbsRel"] for _, r, _ in stages]
    unseen = [mb(k, "AbsRel", False) for _, _, k in stages]
    large = (TEST[2]["macro_without_hypersim"]["AbsRel"], mb("Large, untouched", "AbsRel", False))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True, gridspec_kw={"hspace": 0.28})
    for ax, series, ref, title in ((axes[0], test, large[0], "Frozen test set: AbsRel at each development stage (lower is better)"),
                                   (axes[1], unseen, large[1], "Middlebury 2014 (unseen cameras): AbsRel at the same stages")):
        ax.axhline(ref, color=LARGE, linewidth=2, linestyle=(0, (4, 3)))
        ax.text(len(stages) + 0.2, ref * 1.03, f"Large, untouched  {ref:.3f}", color=INK, fontsize=9, va="bottom", ha="right")
        ax.plot(range(len(stages)), series, color=OURS, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        for i, v in enumerate(series):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
        ax.set_ylim(0, max(max(series), ref) * 1.25)
        ax.set_xlim(-0.4, len(stages) + 0.25)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=8)
        clean(ax)
    axes[1].set_xticks(range(len(stages)), [s for s, _, _ in stages], fontsize=8.5, color=INK)
    handles = [plt.Line2D([], [], color=OURS, marker="o", linewidth=2), plt.Line2D([], [], color=LARGE, linewidth=2, linestyle=(0, (4, 3)))]
    fig.legend(handles, ["Small backbone, each stage (one seed)", "Large, untouched"], loc="upper left", bbox_to_anchor=(0.05, 1.0),
               ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(top=0.9, left=0.055, right=0.985, bottom=0.1)
    fig.savefig(OUT / "progress.png", dpi=160)


def per_source() -> None:
    names = {"bonn": "Bonn RGB-D", "nyu": "NYU Depth V2", "arkit": "ARKitScenes", "diode": "DIODE indoor", "tum_indoor": "TUM freiburg1\n(camera never trained on)"}
    values = {label: [t["per_source"][s]["AbsRel"] for t in TEST] for s, label in names.items()}
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    grouped(ax, values, "lower", 3)
    ax.set_title("Frozen test set by source: AbsRel, lower is better", loc="left", fontsize=12, fontweight="bold", pad=36)
    legend(fig, 0.9)
    fig.subplots_adjust(top=0.77, left=0.055, right=0.985, bottom=0.14)
    fig.savefig(OUT / "per_source.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (headline, parameters, scale_bias, progress, per_source):
        draw()
        print("drew", draw.__name__)
