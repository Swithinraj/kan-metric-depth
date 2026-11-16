# Unseen-data comparison: base Small, ours, Large

None of the three models trained on any image below. Ours is the adapters + votes model trained with the extra ARKitScenes rooms (`runs/split9_kan_data`, one seed). These sets were looked at during development, so they are unseen data, not untouched data.

Parameters: Small (base, untouched) 24.8M; Ours (Small + FastKAN) about 25M; Large (untouched) 335M.

Scale-fixed (shape) columns rescale each prediction by its median before scoring, so they measure relative depth only.

The two TUM sequences of the fresh set (freiburg2 desk, freiburg3 long office) are left out of this comparison; their scores remain in `results/fresh_eval.json`.

## 7-Scenes (200 images: heads and stairs)

### Both sequences (mean of sequences)

| Metric | Small (base, untouched) | Ours (Small + FastKAN) | Large (untouched) |
|---|---|---|---|
| AbsRel (lower) | 0.610 | **0.311** | 0.494 |
| RMSE m (lower) | 0.711 | **0.402** | 0.667 |
| delta1 (higher) | 0.244 | **0.484** | 0.165 |
| bias (1 = right) | 1.571 | **1.254** | 1.449 |
| shape AbsRel (lower) | 0.121 | 0.105 | **0.104** |
| shape delta1 (higher) | 0.834 | 0.882 | **0.893** |

### 7scenes_heads

| Metric | Small (base, untouched) | Ours (Small + FastKAN) | Large (untouched) |
|---|---|---|---|
| AbsRel (lower) | 0.797 | **0.458** | 0.586 |
| RMSE m (lower) | 0.620 | **0.380** | 0.484 |
| delta1 (higher) | 0.085 | **0.221** | 0.057 |
| bias (1 = right) | 1.739 | **1.419** | 1.530 |
| shape AbsRel (lower) | 0.121 | 0.104 | **0.086** |
| shape delta1 (higher) | 0.846 | 0.910 | **0.932** |

### 7scenes_stairs

| Metric | Small (base, untouched) | Ours (Small + FastKAN) | Large (untouched) |
|---|---|---|---|
| AbsRel (lower) | 0.422 | **0.163** | 0.402 |
| RMSE m (lower) | 0.801 | **0.424** | 0.849 |
| delta1 (higher) | 0.402 | **0.748** | 0.272 |
| bias (1 = right) | 1.403 | **1.089** | 1.368 |
| shape AbsRel (lower) | 0.121 | **0.107** | 0.122 |
| shape delta1 (higher) | 0.821 | **0.855** | 0.853 |

## Middlebury stereo (structured-light ground truth)

### Middlebury 2014, DSLR (23 scenes)

| Metric | Small (base, untouched) | Ours (Small + FastKAN) | Large (untouched) |
|---|---|---|---|
| AbsRel (lower) | 0.373 | **0.266** | 0.396 |
| RMSE m (lower) | 1.193 | **0.890** | 1.208 |
| delta1 (higher) | 0.199 | **0.385** | 0.132 |
| bias (1 = right) | 0.643 | **0.820** | 0.618 |
| shape AbsRel (lower) | 0.106 | **0.079** | 0.113 |
| shape delta1 (higher) | 0.849 | **0.929** | 0.891 |

### Middlebury 2021, close range (11 scenes)

| Metric | Small (base, untouched) | Ours (Small + FastKAN) | Large (untouched) |
|---|---|---|---|
| AbsRel (lower) | 0.421 | 0.246 | **0.211** |
| RMSE m (lower) | 0.498 | 0.321 | **0.283** |
| delta1 (higher) | 0.518 | 0.635 | **0.690** |
| bias (1 = right) | 1.370 | 1.227 | **1.101** |
| shape AbsRel (lower) | 0.099 | 0.077 | **0.054** |
| shape delta1 (higher) | 0.904 | 0.939 | **0.989** |
