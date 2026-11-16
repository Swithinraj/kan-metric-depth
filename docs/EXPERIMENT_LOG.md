# Experiment log

Chronological record of every experiment, including the ones that did not help. Section 1-7 cover the frozen test set and the untouched Depth Anything V2 reference models; the new model starts at section 8. The summary is in the [README](../README.md).

Date: 2026-11-14. Raw numbers: `results/depth_anything_v2s_baseline.json`. Reproduce: `python -m tools.make_test_set` then `python -m tools.eval_relative_baseline`.

## 1. The test set

440 images, kept aside before any new model work. Its list is frozen in `datasets/test_set/manifest.json`, and `tools/eval_relative_baseline.py` refuses to run if the list the loader produces no longer matches it.

| Source | Images | Test scenes | Median GT depth | Camera (fx, fy at native res.) | Depth type |
|---|---|---|---|---|---|
| Bonn RGB-D Dynamic | 100 | 6 sequences | 2.38 m | 542.8, 542.6 (Asus Xtion) | structured light, undistorted by us |
| NYU Depth V2 | 81 | 1 (frames) | 2.61 m | 518.9, 519.5 (Kinect) | structured light |
| ARKitScenes (upsampling) | 78 | 3 videos | 1.30 m | ~531-537 at 640x480 (iPad LiDAR) | LiDAR, 1920x1440 |
| DIODE indoor | 73 | 2 scans | 1.79 m | 886.8, 927.1 | laser scanner |
| Hypersim | 8 | 2 scenes | 4.95 m | 886.8, 886.8 | synthetic, exact |
| TUM RGB-D freiburg1 | 100 | 2 sequences (desk, room) | 1.27 m | 517.3, 516.5 (Kinect) | structured light, undistorted by us |

How it was chosen:
- A scene is held out whole. Within each source the scenes are sorted; every 4th goes to test, a different every-4th set goes to validation, the rest train. NYU is split by frame (every 8th). TUM freiburg1 is test-only and never trained on.
- Each source is capped at 100 images, picked evenly spaced across its test pool, so one large source (Bonn) cannot dominate.
- `tools/make_test_set.py` asserts that no test frame and no test scene appears in train or validation. It passes: 0 shared frames, 0 shared scenes.
- Rule going forward: do not train on it, tune on it, or pick checkpoints with it. Use the validation split for that.

Inputs are the loader's 252x336 resize (multiple of 14), depth valid in 0.1-20 m.

## 2. Protocol

Depth Anything V2-Small (`depth-anything/Depth-Anything-V2-Small-hf`, 24,785,089 parameters, frozen, as downloaded) outputs relative disparity, not meters. Every image is therefore aligned to ground truth before scoring, which means these numbers measure depth shape, not metric ability:

- Affine (standard): per-image least-squares scale and shift in disparity space.
- Scale-only: per-image scale only, no shift. Stricter, because zero disparity must mean infinite depth.

Metrics, over valid pixels: AbsRel (lower is better), RMSE in meters (lower), delta1 = share of pixels within a 1.25 ratio of ground truth (higher). "Macro" averages the six source means; "micro" averages all 440 images.

## 3. Results (base Depth Anything V2-Small, run on an RTX 3050 laptop GPU, 36 s)

Affine alignment:

| Source | AbsRel | RMSE (m) | delta1 |
|---|---|---|---|
| Bonn | 0.050 | 0.170 | 0.987 |
| NYU | 0.104 | 0.458 | 0.903 |
| ARKitScenes | 0.068 | 0.124 | 0.959 |
| DIODE | 0.099 | 0.476 | 0.916 |
| Hypersim | 0.087 | 1.155 | 0.911 |
| TUM freiburg1 | 0.060 | 0.177 | 0.962 |
| **Macro** | **0.078** | **0.427** | **0.940** |
| Micro | 0.074 | 0.285 | 0.948 |

Scale-only alignment:

| Source | AbsRel | RMSE (m) | delta1 |
|---|---|---|---|
| Bonn | 0.546 | 2.841 | 0.509 |
| NYU | 0.783 | 4.381 | 0.348 |
| ARKitScenes | 0.497 | 1.441 | 0.550 |
| DIODE | 0.451 | 2.087 | 0.565 |
| Hypersim | 0.263 | 3.357 | 0.733 |
| TUM freiburg1 | 0.385 | 1.764 | 0.657 |
| **Macro** | **0.487** | **2.645** | **0.560** |
| Micro | 0.523 | 2.516 | 0.534 |

## 4. What this shows

- Depth shape is already strong: with a per-image scale and shift, macro delta1 is 0.94. Bonn and TUM are easiest (AbsRel 0.050 and 0.060); NYU and DIODE are hardest (0.104 and 0.099).
- The scale-only collapse (macro AbsRel 0.49) shows its disparity is offset by an unknown per-image shift. Recovering real meters therefore means estimating both scale and shift, and nothing in the base model does that. This is the gap the intrinsics-aware metric head has to close.
- These are the numbers any new model must be compared against. A fair target for a metric model is to approach the affine row without being given ground truth.

## 5. Caveats

- Ground-truth alignment is used, so none of this is a metric result. A metric model scored without alignment is held to a harder standard than this baseline.
- Images are scored at 252x336, well below the model's native resolution, so absolute numbers are not comparable to published benchmarks (our NYU subset is 81 frames, not the standard 654).
- Hypersim has only 8 test images from 2 scenes; treat its row as indicative. Its RMSE is inflated by depths of 5 m and more.
- Hypersim scenes with tilted lenses were excluded, and Bonn and TUM images were undistorted (see the intrinsics checks in the loader notes).
- Section 6 covers Depth Anything V2's metric indoor checkpoint, which was fine-tuned on Hypersim, so its Hypersim row is contaminated.

## 6. Metric depth (raw output in meters, no alignment to ground truth)

Model: `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (24,785,089 parameters, max depth 20 m), the official metric fine-tune of the same Small backbone, run frozen on the same 440 images. Raw predictions are scored directly against ground-truth meters. `bias` is the median of prediction / ground truth per image (1.0 = correct scale, above 1 = predicted too far). The last two columns repeat AbsRel and delta1 after a per-image median-scale correction, as a shape-only reference. Raw numbers: `results/depth_anything_v2_metric_indoor_small_metric.json`.

| Source | AbsRel | RMSE (m) | delta1 | bias | AbsRel, scale-fixed | delta1, scale-fixed |
|---|---|---|---|---|---|---|
| Bonn | 0.354 | 0.721 | 0.505 | 1.34 | 0.065 | 0.965 |
| NYU | 0.341 | 0.936 | 0.384 | 1.32 | 0.090 | 0.912 |
| ARKitScenes | 0.500 | 0.564 | 0.347 | 1.41 | 0.100 | 0.891 |
| DIODE | 0.257 | 0.542 | 0.599 | 1.16 | 0.108 | 0.870 |
| Hypersim (contaminated) | 0.146 | 1.681 | 0.763 | 0.99 | 0.097 | 0.896 |
| TUM freiburg1 | 0.516 | 0.683 | 0.195 | 1.47 | 0.112 | 0.866 |
| **Macro, all six** | **0.352** | **0.854** | **0.465** | 1.28 | 0.095 | 0.900 |
| **Macro, without Hypersim** | **0.394** | **0.689** | **0.406** | 1.34 | 0.095 | 0.901 |
| Micro | 0.395 | 0.712 | 0.405 | 1.34 | 0.094 | 0.903 |

What this shows:
- Metric depth is the weak point. Without Hypersim, only about 41% of pixels land within 25% of the true depth (delta1 0.406), and AbsRel is 0.394, against 0.078 and 0.940 when shape alone is scored with a fitted scale and shift.
- The error is mostly scale, not shape. After a per-image median-scale fix delta1 returns to 0.90.
- The scale error is systematic: the model predicts 16-47% too far on every real-camera source. It is accurate only on Hypersim (bias 0.99), the data it was fine-tuned on, so that row says little about generalization.
- Real-camera sources with the highest bias (TUM 1.47, ARKit 1.41, Bonn 1.34, NYU 1.32) are Kinect-class or iPad cameras at 520-540 px focal length per 640 px width; the lowest real-camera bias (DIODE 1.16) has the same field of view as Hypersim. That is consistent with camera dependence, but the focal-length gap between them is only about 7%, far smaller than the bias, so it cannot be the whole story: synthetic-to-real sensor and scene differences are likely contributing too.
- Inputs here are 252x336, upsampled by the processor; the resolution may also affect the scale the model infers. This was not isolated.

Target for the new model: close the metric gap on the real-camera sources (bias near 1.0, delta1 well above 0.41), without giving up the shape quality in sections 3 and 4.

## 7. Reference: metric indoor Small vs Base vs Large (raw meters, no alignment)

Same 440 test images, same protocol as section 6. Macro over the five sources other than Hypersim (all three checkpoints were fine-tuned on Hypersim). Raw numbers: `results/depth_anything_v2_metric_indoor_{small,base,large}_metric.json`.

| Model | Parameters | AbsRel | RMSE (m) | delta1 | bias | AbsRel, scale-fixed | delta1, scale-fixed |
|---|---|---|---|---|---|---|---|
| Small | 24.8M | 0.394 | 0.689 | 0.406 | 1.34 | 0.095 | 0.901 |
| Base | 97.5M | 0.336 | 0.617 | 0.454 | 1.28 | 0.079 | 0.925 |
| Large | 335.3M | 0.311 | 0.580 | 0.494 | 1.26 | 0.070 | 0.941 |

Per-source bias (Small / Base / Large): Bonn 1.34 / 1.31 / 1.23, NYU 1.32 / 1.29 / 1.28, ARKitScenes 1.41 / 1.38 / 1.37, DIODE 1.16 / 1.08 / 1.08, TUM freiburg1 1.47 / 1.35 / 1.34.

- Model size improves shape (scale-fixed delta1 0.90 -> 0.94) but barely moves the scale error: Large still predicts about 26% too far and gets under half the pixels within 25%.
- The number to beat with the Small backbone is Large's raw result: AbsRel 0.311, delta1 0.494, bias 1.26.

Correction on resolution: the official processor resizes our 252x336 inputs to 392x518, and that is the resolution behind sections 6 and 7. Working resolution changes the metric scale a lot, so section 8 re-scores all three checkpoints at 518x686 and uses those as the reference.

## 8. Intrinsics-conditioned model (Small backbone) vs references, all at 518x686

Model: `kan_metric_depth/model.py`, trained with `train.py`. Metric indoor Small checkpoint, backbone frozen, neck and head fine-tuned (2.73M parameters). Camera code per location (ray x, ray y, log focal x, log focal y) drives FiLM on the four backbone taps and four fusion outputs plus a global log-scale. Three runs, identical except for the conditioner: none (plain fine-tune control), MLP (527,859 new parameters), FastKAN (526,407). 8 epochs of 3,000 samples, batch 4, focal-length augmentation (zoom 0.7-2.0), L1 on log-depth in meters plus gradient matching, checkpoint picked on validation. One seed each. References are the untouched checkpoints run through the same code path (`--init-only`). Raw numbers: `results/runs/intr_{none,mlp,kan}_test.json`, `results/runs/{init_518,ref_Base_518,ref_Large_518}_init_test.json`.

Frozen test set, raw meters, macro over the five real-camera sources:

| Model | Total parameters | AbsRel | RMSE (m) | delta1 | bias | AbsRel, scale-fixed | delta1, scale-fixed |
|---|---|---|---|---|---|---|---|
| Small, untouched | 24.8M | 0.303 | 0.560 | 0.536 | 1.20 | 0.100 | 0.885 |
| Base, untouched | 97.5M | 0.271 | 0.521 | 0.604 | 1.18 | 0.081 | 0.923 |
| Large, untouched | 335.3M | 0.246 | 0.477 | 0.629 | 1.17 | 0.068 | 0.940 |
| Small + fine-tune, no intrinsics | 24.8M | 0.242 | 0.450 | 0.641 | 1.09 | 0.115 | 0.870 |
| Small + MLP intrinsics | 25.3M | 0.219 | 0.411 | 0.679 | 1.06 | 0.101 | 0.894 |
| **Small + FastKAN intrinsics** | 25.3M | **0.207** | **0.409** | **0.686** | **1.03** | 0.099 | 0.898 |

Per source, AbsRel / delta1 / bias:

| Source | Large, untouched | No intrinsics | MLP | FastKAN |
|---|---|---|---|---|
| Bonn | 0.172 / 0.890 / 1.15 | 0.159 / 0.770 / 1.01 | 0.158 / 0.810 / 1.04 | 0.151 / 0.817 / 1.03 |
| NYU | 0.229 / 0.630 / 1.20 | 0.143 / 0.805 / 0.99 | 0.124 / 0.860 / 1.03 | 0.125 / 0.861 / 1.03 |
| ARKitScenes | 0.357 / 0.470 / 1.27 | 0.382 / 0.466 / 1.21 | 0.365 / 0.426 / 1.09 | 0.351 / 0.433 / 1.07 |
| DIODE | 0.203 / 0.659 / 1.00 | 0.290 / 0.559 / 1.11 | 0.243 / 0.640 / 1.09 | 0.225 / 0.626 / 1.00 |
| TUM freiburg1 (never trained on) | 0.269 / 0.498 / 1.25 | 0.233 / 0.603 / 1.11 | 0.203 / 0.657 / 1.07 | 0.185 / 0.693 / 1.02 |
| Hypersim (8 images) | 0.165 / 0.686 / 0.93 | 0.217 / 0.537 / 0.92 | 0.220 / 0.452 / 0.86 | 0.220 / 0.462 / 0.86 |

Intrinsics-use check (same test images, only the fx, fy given to the model are multiplied): x0.75 gives predicted depth x0.762 (MLP) and x0.772 (FastKAN); x1.5 gives x1.330 and x1.324. The ideal is x0.75 and x1.5, so both conditioned models have learned that depth scales with focal length.

Best validation AbsRel: no intrinsics 0.187 (epoch 4), MLP 0.161 (epoch 6), FastKAN 0.173 (epoch 7).

What this shows:
- The FastKAN model beats untouched Large on the raw metric macro (AbsRel 0.207 vs 0.246, delta1 0.686 vs 0.629) and has the bias closest to 1 (1.03 vs 1.17), with 13 times fewer parameters.
- Intrinsics help beyond plain fine-tuning: AbsRel 0.242 -> 0.219 (MLP) -> 0.207 (FastKAN). The largest gain is on TUM, the source never trained on.
- FastKAN vs MLP is not settled. FastKAN is ahead on test (0.207 vs 0.219) but behind on validation (0.173 vs 0.161), with one seed each. The gap is within what seed noise could explain.
- Large is still better where shape matters: scale-fixed delta1 0.940 vs 0.898, and raw delta1 on Bonn, ARKitScenes and DIODE. Plain fine-tuning actually cost the Small model some shape (0.885 -> 0.870); the conditioned runs recover it.
- Hypersim got worse in every fine-tuned run (16 training images, and the checkpoint started out fitted to it).

Caveats:
- This is not a like-for-like win over Large. Our models were fine-tuned on other scenes from Bonn, NYU, ARKitScenes and DIODE; Large was run as downloaded. Only TUM is unseen by both, and even there the camera is close to NYU's Kinect.
- All real cameras here have nearly the same field of view, so the focal-length behaviour was learned from zoom augmentation and is only tested by the synthetic check above, not by a genuinely different real lens.
- Single seed, 440 test images, ARKitScenes test covers 3 videos.

## 9. FastKAN vs MLP over three seeds

Seeds 0 (the section 8 runs), 1 and 2, otherwise identical (`train.py --seed`). Mean ± sample standard deviation over the three seeds; test numbers are the macro over the five real-camera sources. Raw numbers: `runs/intr_{mlp,kan}`, `runs/intr_{mlp,kan}_s1`, `runs/intr_{mlp,kan}_s2`.

| Measure | MLP | FastKAN |
|---|---|---|
| Test AbsRel | 0.213 ± 0.006 (0.219, 0.214, 0.206) | 0.205 ± 0.002 (0.207, 0.204, 0.205) |
| Test RMSE (m) | 0.409 ± 0.003 | 0.408 ± 0.001 |
| Test delta1 | 0.686 ± 0.007 | 0.684 ± 0.002 |
| Test bias | 1.050 ± 0.012 | 1.024 ± 0.008 |
| Test delta1, scale-fixed | 0.896 ± 0.002 | 0.898 ± 0.001 |
| Best validation AbsRel | 0.163 ± 0.002 | 0.171 ± 0.002 |
| Depth change for focal x0.75 (ideal 0.75) | 0.760 ± 0.018 | 0.780 ± 0.011 |
| Depth change for focal x1.5 (ideal 1.5) | 1.348 ± 0.039 | 1.344 ± 0.025 |

Test AbsRel per source (MLP / FastKAN, means): Bonn 0.156 / 0.151, NYU 0.123 / 0.123, ARKitScenes 0.351 / 0.340, DIODE 0.242 / 0.218, TUM freiburg1 0.194 / 0.195.

- On test, FastKAN has a slightly lower AbsRel and a scale closer to 1 in every seed, but the AbsRel gap (0.008) is about the size of the MLP's own seed spread, and MLP seed 2 (0.206) reaches FastKAN's level. RMSE, delta1 and shape are ties.
- Most of FastKAN's test edge comes from DIODE (0.218 vs 0.242, consistent in all three seeds), the one real camera with a different focal length. TUM, the unseen source, is a tie.
- On validation the MLP is ahead in every seed (0.163 vs 0.171), so the two held-out splits disagree about which is better.
- Both use the intrinsics about equally: neither is closer to the ideal focal-length response.
- Conclusion: intrinsics conditioning is a solid gain over plain fine-tuning (0.242 -> about 0.21, all six conditioned runs). A FastKAN advantage over a size-matched MLP is not demonstrated by this data.

## 10. Unseen dataset, unseen cameras: Middlebury stereo 2014 and 2021

Neither our models nor Depth Anything's metric checkpoints were trained on this data, so it is the like-for-like comparison section 8 lacked. Evaluation only (`eval_middlebury.py`, data in `datasets/middlebury`, raw rows in `results/unseen_middlebury.json`); nothing was trained, tuned or selected on it. One left image per scene, centre-cropped to 4:3, same 252x336 input, 518x686 working size and raw-meter scoring as before. Ground truth is structured-light disparity converted with Z = baseline x f / (disparity + doffs) from each scene's `calib.txt`. Our MLP and FastKAN rows are the mean of the three seeds.

iBims-1 was the first choice but its download sits behind a login and bot protection, so it was not used.

Middlebury 2014: 23 scenes, DSLR, 1.5-6 m, focal length per image width 0.95-3.13 after the crop (training cameras are about 0.83; zoom augmentation covered roughly 0.6-1.7).

| Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed |
|---|---|---|---|---|---|
| Small, untouched | 0.373 | 1.193 | 0.199 | 0.64 | 0.849 |
| Base, untouched | 0.381 | 1.212 | 0.193 | 0.64 | 0.862 |
| Large, untouched | 0.396 | 1.208 | 0.132 | 0.62 | 0.891 |
| Small + fine-tune, no intrinsics | 0.283 | 0.954 | 0.346 | 0.73 | 0.873 |
| Small + MLP intrinsics | 0.205 ± 0.014 | 0.691 | 0.657 | 1.05 | 0.926 |
| Small + FastKAN intrinsics | 0.197 ± 0.010 | 0.699 | 0.672 | 1.01 | 0.911 |

Split by lens (AbsRel / delta1 / bias): 11 scenes below 1.7 focal-per-width: Large 0.275 / 0.241 / 0.78, MLP 0.163 / 0.789 / 1.03, FastKAN 0.155 / 0.812 / 1.04. 12 scenes at 1.7 and above (beyond the augmentation range): Large 0.507 / 0.033 / 0.47, MLP 0.243 / 0.536 / 1.06, FastKAN 0.236 / 0.543 / 0.99.

Middlebury 2021, 11 close-range scenes (chess, curule, skates, skiboots; iPod touch, 0.7-1.6 m, focal-per-width 1.22):

| Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed |
|---|---|---|---|---|---|
| Small, untouched | 0.421 | 0.498 | 0.518 | 1.37 | 0.904 |
| Base, untouched | 0.252 | 0.322 | 0.699 | 1.16 | 0.936 |
| Large, untouched | 0.211 | 0.283 | 0.690 | 1.10 | 0.989 |
| Small + fine-tune, no intrinsics | 0.383 | 0.448 | 0.503 | 1.38 | 0.887 |
| Small + MLP intrinsics | 0.630 ± 0.048 | 0.723 | 0.282 | 1.69 | 0.890 |
| Small + FastKAN intrinsics | 0.617 ± 0.009 | 0.721 | 0.281 | 1.68 | 0.889 |

Middlebury 2021, the other 13 scenes (artroom, bandsaw, ladder, octogons, pendulum, podium, traproom) are not counted. Their calibration gives depths of 7-14 m for what look like ordinary rooms, several have principal points far outside the image, and every model, including untouched Large, predicts about a quarter of that (bias 0.24-0.30, delta1 0.000 for all). The ground-truth scale for these is doubtful; this was decided from the calibration and the untouched models, and our models score about the same as Large on them (AbsRel 0.69-0.70 vs 0.76).

What this shows:
- On the unseen DSLR scenes, the conditioned models are far ahead of Large (AbsRel about 0.20 vs 0.40, delta1 0.66 vs 0.13) and their scale is right on average (bias 1.01-1.05 vs 0.62). Large gets worse as the lens gets longer; ours holds even beyond the focal lengths seen in augmentation. This is the first real-lens evidence that the intrinsics conditioning works, and plain fine-tuning without intrinsics does not get it (0.283, bias 0.73).
- On the close-range iPod scenes the result reverses: Large is much better (AbsRel 0.21 vs 0.62) and our conditioned models predict about 70% too far. They are also worse than our own no-intrinsics control (0.383) and the untouched Small (0.421), so here the conditioning hurts. The likely cause: training only ever paired a longer focal length with a zoomed-in room, so the model learned "longer lens means farther" and has never seen an object genuinely 1 m from a longer lens. This is an inference, not tested.
- FastKAN vs MLP: FastKAN is nominally ahead on both subsets but within the seed spread on each. Still not demonstrated.
- Sample sizes are small (23 and 11 single images), and the 2014 scenes are staged still lifes, not typical rooms.

## 11. Split shape / scale model (no distillation, no new data)

Model: `SplitScaleDepth` in `kan_metric_depth/model.py`, trained with `train.py --arch split`. The decoder's log-depth has its per-image mean removed and is trained with a scale-blind loss (median-centred L1 on log-depth plus gradient matching). One scale branch reads the camera (2 log focal lengths, principal point) together with content (class token and mean patch token of the four backbone taps, projected to 32 numbers, plus the decoder's own level and spread, gradients blocked) and outputs a log-scale and a gain on the shape. The metric L1 loss sees a detached shape, so its gradient reaches the branch only (checked: exactly zero into the decoder, and zero from the shape loss into the branch). The eight camera-only FiLM conditioners are gone; new parameters 110,018 (was 526,407). Same data, schedule and 518x686 working size as section 8. One seed each. Raw numbers: `results/runs/split_{kan,mlp,kan_nocam}_test.json`, `results/unseen_middlebury.json`.

Frozen test set, raw meters, macro over the five real-camera sources:

| Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed | best val AbsRel |
|---|---|---|---|---|---|---|
| Large, untouched | 0.246 | 0.477 | 0.629 | 1.17 | 0.940 | - |
| FiLM + FastKAN (section 9, 3 seeds) | 0.205 | 0.408 | 0.684 | 1.02 | 0.898 | 0.171 |
| Split + FastKAN | 0.226 | 0.401 | 0.658 | 1.07 | 0.909 | 0.164 |
| Split + MLP | 0.235 | 0.407 | 0.651 | 1.09 | 0.910 | 0.166 |
| Split + FastKAN, camera hidden | 0.251 | 0.433 | 0.631 | 1.11 | 0.912 | 0.174 |

Per-source AbsRel, Split + FastKAN (FiLM + FastKAN seed mean in brackets): Bonn 0.152 (0.151), NYU 0.113 (0.123), ARKitScenes 0.339 (0.340), DIODE 0.269 (0.218), TUM freiburg1 0.255 (0.195), Hypersim 0.145 (0.220). Focal check for Split + FastKAN: x0.75 -> x0.879, x1.5 -> x1.277 (FiLM: x0.78, x1.34).

Middlebury (unseen; AbsRel / delta1 / bias / delta1 scale-fixed):

| Model | 2014, all 23 | 2014, focal-per-width >= 1.7 (12) | 2021 close-range (11) |
|---|---|---|---|
| Large, untouched | 0.396 / 0.132 / 0.62 / 0.891 | 0.507 / 0.033 / 0.47 / 0.824 | 0.211 / 0.690 / 1.10 / 0.989 |
| FiLM + FastKAN, seed 0 | 0.199 / 0.657 / 0.99 / 0.897 | 0.245 / 0.497 / 0.96 / 0.887 | 0.617 / 0.272 / 1.66 / 0.889 |
| Split + FastKAN | 0.213 / 0.629 / 1.02 / 0.933 | 0.186 / 0.616 / 0.94 / 0.948 | 0.393 / 0.520 / 1.43 / 0.909 |
| Split + MLP | 0.216 / 0.690 / 1.11 / 0.942 | 0.220 / 0.686 / 1.13 / 0.963 | 0.482 / 0.422 / 1.52 / 0.917 |
| Split + FastKAN, camera hidden | 0.337 / 0.297 / 0.68 / 0.911 | 0.443 / 0.086 / 0.55 / 0.898 | 0.251 / 0.652 / 1.19 / 0.914 |

What this shows:
- Shape improved everywhere: test 0.898 -> 0.909, validation 0.928 -> 0.935, Middlebury 2014 0.897 -> 0.933 (now above Large's 0.891 there). Freeing the decoder from the metric loss helps it.
- Close-range error fell by about a third (AbsRel 0.617 -> 0.393) but Large is still clearly better (0.211).
- In-domain metric accuracy slipped (test AbsRel 0.205 -> 0.226), almost entirely on TUM and DIODE. Still ahead of Large (0.246).
- The camera input is doing real work on long lenses (2014: 0.213 with camera vs 0.337 without) and is what hurts on close-range (0.393 with vs 0.251 without). The content-only branch nearly matches Large on close-range but fails on long lenses. So the branch has not learned the interaction that was the point of feeding both: it applies the focal correction regardless of content.
- FastKAN vs MLP: FastKAN ahead on test, validation and close-range, MLP ahead on Middlebury 2014 delta1. One seed; not settled.
- Middlebury has now been looked at twice and informed this design, so it is a development set, not a clean test.

## 12. Split model with more cameras and wider augmentation

Added to training and validation only: SUN RGB-D without its Kinect v1 part (that part contains NYU, including frozen-test frames). Three cameras, intrinsics read from each image's `intrinsics.txt`: Kinect v2 (focal per width 0.725, 819 training images), RealSense (1.02, 779), Xtion (0.965, 774); existing cameras are near 0.83. Depth is SUN RGB-D's hole-filled `depth_bfx`, so some values are interpolated. Scenes are rooms at about 2 m (RealSense included), so this adds lenses but no close-range data. Training is now 5,978 images, validation 565; the frozen test set is identical to its manifest.

Augmentation changes: zoom range 0.5-3.0 (was 0.7-2.0); working size drawn per step from 392x518, 434x574, 476x630, 518x686 with intrinsics unchanged. 12 epochs (was 8). Otherwise as section 11. One seed each. Runs: `runs/split2_{kan,mlp}`. Validation numbers are not comparable with earlier sections because validation now includes the new cameras.

Frozen test set, raw meters, macro over the five real-camera sources:

| Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed |
|---|---|---|---|---|---|
| Large, untouched | 0.246 | 0.477 | 0.629 | 1.17 | 0.940 |
| FiLM + FastKAN (section 9, 3 seeds) | 0.205 | 0.408 | 0.684 | 1.02 | 0.898 |
| Split + FastKAN (section 11) | 0.226 | 0.401 | 0.658 | 1.07 | 0.909 |
| Split + FastKAN, more data + augmentation | 0.209 | 0.395 | 0.678 | 1.03 | 0.909 |
| Split + MLP, more data + augmentation | 0.219 | 0.403 | 0.667 | 1.05 | 0.909 |

Per-source AbsRel / delta1 / bias, Split + FastKAN with more data: Bonn 0.147 / 0.813 / 1.01, NYU 0.112 / 0.876 / 0.99, ARKitScenes 0.339 / 0.440 / 1.04, DIODE 0.247 / 0.574 / 1.02, TUM freiburg1 0.201 / 0.690 / 1.12, Hypersim 0.240 / 0.397 / 0.80. Focal check: x0.75 -> x0.880, x1.5 -> x1.251.

Middlebury (development set; AbsRel / delta1 / bias / delta1 scale-fixed):

| Model | 2014, all 23 | 2014, focal-per-width < 1.7 (11) | 2014, >= 1.7 (12) | 2021 close-range (11) |
|---|---|---|---|---|
| Large, untouched | 0.396 / 0.132 / 0.62 / 0.891 | 0.275 / 0.241 / 0.78 / 0.965 | 0.507 / 0.033 / 0.47 / 0.824 | 0.211 / 0.690 / 1.10 / 0.989 |
| Split + FastKAN (section 11) | 0.213 / 0.629 / 1.02 / 0.933 | 0.242 / 0.642 / 1.11 / 0.916 | 0.186 / 0.616 / 0.94 / 0.948 | 0.393 / 0.520 / 1.43 / 0.909 |
| Split + FastKAN, more data + aug | 0.199 / 0.622 / 1.01 / 0.939 | 0.146 / 0.810 / 1.01 / 0.924 | 0.247 / 0.450 / 1.01 / 0.953 | 0.420 / 0.456 / 1.45 / 0.918 |
| Split + MLP, more data + aug | 0.197 / 0.598 / 1.03 / 0.945 | 0.184 / 0.673 / 1.05 / 0.926 | 0.209 / 0.528 / 1.01 / 0.963 | 0.444 / 0.432 / 1.47 / 0.911 |

What this shows:
- The in-domain metric loss from section 11 is recovered (test AbsRel 0.226 -> 0.209, TUM 0.255 -> 0.201) while keeping the better shape, so this is the best model so far on the frozen test set taken as a whole: metric about equal to the FiLM model, shape better, RMSE lowest.
- Close-range is unchanged (AbsRel 0.42, bias 1.45; Large 0.21). Neither the wider zoom nor the resolution jitter moved it, and the new data has no close-range scenes. This remains the one place Large clearly wins.
- Shape on the test set did not move (0.909 vs Large 0.940).
- Hypersim got worse (delta1 0.745 -> 0.397, bias 0.80). It has 16 training images among 5,978 and 8 test images.
- The camera response is still sub-linear (x0.88 / x1.25 for x0.75 / x1.5).
- FastKAN vs MLP: FastKAN ahead on the test set and normal lenses, MLP on long lenses. One seed; not settled.

## 13. Split + FastKAN with close-range data added

Added to training and validation only: six real-sensor tabletop sets from the BOP benchmark collection (`datasets/bop`, 3.3 GB), intrinsics and depth scale read from each `scene_camera.json`. Focal per width / typical depth: YCB-Video 1.67 / 0.94 m (637 images used), T-LESS 1.49 / 0.78 m (693), Linemod-Occluded 0.89 / 1.04 m (223), HOPE 0.72 / 0.72 m (139), HomebrewedDB 0.84 / 1.29 m (222), TUD-L 0.80 / 3.3 m (222). They enter as two sources (long-lens: YCB-Video and T-LESS; ordinary-lens: the rest), about a fifth of each epoch. Training is 7,954 images, validation 725; the frozen test set is identical to its manifest. Otherwise as section 12, 12 epochs of 3,500 samples, FastKAN only, one seed. Run: `runs/split3_kan`.

Frozen test set, raw meters, macro over the five real-camera sources:

| Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed |
|---|---|---|---|---|---|
| Large, untouched | 0.246 | 0.477 | 0.629 | 1.17 | 0.940 |
| Split + FastKAN, section 12 | 0.209 | 0.395 | 0.678 | 1.03 | 0.909 |
| Split + FastKAN, + close-range data | 0.213 | 0.407 | 0.668 | 1.03 | 0.905 |

Per-source AbsRel / delta1 / bias: Bonn 0.163 / 0.790 / 1.01, NYU 0.113 / 0.875 / 0.98, ARKitScenes 0.326 / 0.435 / 1.02, DIODE 0.267 / 0.542 / 1.03, TUM freiburg1 0.194 / 0.697 / 1.11, Hypersim 0.149 / 0.724 / 0.97. Focal check: x0.75 -> x0.892, x1.5 -> x1.233.

Middlebury (development set; AbsRel / delta1 / bias / delta1 scale-fixed):

| Model | 2014, all 23 | 2014, focal-per-width < 1.7 (11) | 2014, >= 1.7 (12) | 2021 close-range (11) |
|---|---|---|---|---|
| Large, untouched | 0.396 / 0.132 / 0.62 / 0.891 | 0.275 / 0.241 / 0.78 / 0.965 | 0.507 / 0.033 / 0.47 / 0.824 | 0.211 / 0.690 / 1.10 / 0.989 |
| Split + FastKAN, section 12 | 0.199 / 0.622 / 1.01 / 0.939 | 0.146 / 0.810 / 1.01 / 0.924 | 0.247 / 0.450 / 1.01 / 0.953 | 0.420 / 0.456 / 1.45 / 0.918 |
| Split + FastKAN, + close-range data | 0.300 / 0.416 / 0.73 / 0.900 | 0.225 / 0.606 / 0.85 / 0.900 | 0.369 / 0.240 / 0.63 / 0.900 | 0.211 / 0.657 / 1.19 / 0.924 |

What this shows:
- Close-range is fixed to Large's level: AbsRel 0.420 -> 0.211 (Large 0.211), delta1 0.657 (Large 0.690). Large still has much better shape there (0.989 vs 0.924).
- The frozen test set is essentially unchanged (0.213 vs 0.209), and Hypersim recovered (delta1 0.397 -> 0.724).
- The long-lens DSLR scenes got clearly worse: AbsRel 0.199 -> 0.300 and the model now predicts 27% too close (bias 0.73). Still ahead of Large (0.396, bias 0.62), but a third of the previous advantage is gone, and shape there fell from 0.939 to 0.900.
- The two results are the same failure seen from the other side. In the added data every long-lens image is a tabletop under 1 m, so the scale branch moved toward "long lens means near" where before it had "long lens means far". It is still tying scale to the lens through whatever the training data pairs with it, not reading the distance from the content.
- The camera response weakened again (x0.89 / x1.23).

## 14. Lens-balanced sampling

Same data and model as section 13. Change: every training image's output focal length per image width is drawn log-uniformly from 0.5-2.8 whatever camera it came from, and the zoom is set to reach it (clipped to 0.5-4.0), instead of a zoom relative to each camera's own lens (`UnifiedDepth(focal_range=...)`, `train.py --focal-min 0.5 --focal-max 2.8 --zoom-max 4.0`). On 900 sampled training images the long-lens tabletop source's focal median went from 2.17 to 1.43 and its 90th percentile from 4.18 to 2.36, against 1.16-1.29 and 2.13-2.48 for the room sources, so long lenses are now shared between near and far scenes. FastKAN, one seed. Run: `runs/split4_kan`.

AbsRel / delta1 / bias / delta1 scale-fixed:

| Model | Frozen test set (5 real cameras) | Middlebury 2021 close-range (11) | Middlebury 2014, all 23 | 2014, focal-per-width < 1.7 (11) | 2014, >= 1.7 (12) |
|---|---|---|---|---|---|
| Large, untouched | 0.246 / 0.629 / 1.17 / 0.940 | 0.211 / 0.690 / 1.10 / 0.989 | 0.396 / 0.132 / 0.62 / 0.891 | 0.275 / 0.241 / 0.78 / 0.965 | 0.507 / 0.033 / 0.47 / 0.824 |
| Section 12 (no close-range data) | 0.209 / 0.678 / 1.03 / 0.909 | 0.420 / 0.456 / 1.45 / 0.918 | 0.199 / 0.622 / 1.01 / 0.939 | 0.146 / 0.810 / 1.01 / 0.924 | 0.247 / 0.450 / 1.01 / 0.953 |
| Section 13 (+ close-range data) | 0.213 / 0.668 / 1.03 / 0.905 | 0.211 / 0.657 / 1.19 / 0.924 | 0.300 / 0.416 / 0.73 / 0.900 | 0.225 / 0.606 / 0.85 / 0.900 | 0.369 / 0.240 / 0.63 / 0.900 |
| Lens-balanced | 0.212 / 0.672 / 1.03 / 0.905 | 0.204 / 0.706 / 1.17 / 0.924 | 0.265 / 0.450 / 0.78 / 0.894 | 0.206 / 0.643 / 0.87 / 0.894 | 0.319 / 0.273 / 0.71 / 0.895 |

Test per source (AbsRel): Bonn 0.155, NYU 0.113, ARKitScenes 0.347, DIODE 0.257, TUM freiburg1 0.188, Hypersim 0.194. Focal check: x0.75 -> x0.892, x1.5 -> x1.242.

What this shows:
- Lens balancing recovered about a third of the DSLR loss (AbsRel 0.300 -> 0.265, bias 0.73 -> 0.78) and kept everything else: close-range is now marginally ahead of Large on both AbsRel and delta1, the test set is unchanged.
- It did not restore section 12's DSLR result (0.199, bias 1.01). The model still places those scenes about 22% too close, and that holds for the ordinary-lens DSLR scenes too (bias 0.87), where the lens cannot be the reason. So the remaining error is a content effect: the DSLR scenes are staged objects at 2-5 m, and the tabletop training data taught the model that object-filled scenes are near.
- Against Large, raw AbsRel is now lower on all three benchmarks (0.212 vs 0.246, 0.204 vs 0.211, 0.265 vs 0.396). The close-range margin is within noise for 11 images. Large keeps the better shape on the test set (0.940 vs 0.905) and on close-range (0.989 vs 0.924).
- One seed. Middlebury is a development set.

## 15. Confirmation: three seeds and a fresh held-out set

Seeds 0 (section 14), 1 and 2 of the lens-balanced Split + FastKAN model (`runs/split4_kan`, `_s1`, `_s2`). Mean ± sample standard deviation.

Fresh held-out set (`eval_fresh.py`, `datasets/fresh_eval`, `results/fresh_eval.json`): never trained on and never used for any decision before this section. 7-Scenes heads and stairs (Kinect, default intrinsics 585/585/320/240, depth not exactly registered to colour) and TUM RGB-D freiburg2 desk and freiburg3 long office (other sensors than the freiburg1 one in the test set, undistorted where needed). 100 evenly spaced frames each, same protocol as the test set. All four cameras are close to the ordinary room lens, so this tests new scenes and sensors, not new lenses.

| Benchmark | Model | AbsRel | RMSE (m) | delta1 | bias | delta1, scale-fixed |
|---|---|---|---|---|---|---|
| Frozen test set, 5 real cameras | Large, untouched | 0.246 | 0.477 | 0.629 | 1.17 | 0.940 |
| | Ours, 3 seeds | 0.208 ± 0.005 | 0.401 ± 0.008 | 0.681 ± 0.012 | 1.03 | 0.906 ± 0.002 |
| Middlebury 2014 DSLR (23) | Large, untouched | 0.396 | 1.208 | 0.132 | 0.62 | 0.891 |
| | Ours, 3 seeds | 0.274 ± 0.010 | 0.865 ± 0.034 | 0.399 ± 0.052 | 0.77 | 0.888 ± 0.009 |
| Middlebury 2021 close-range (11) | Large, untouched | 0.211 | 0.283 | 0.690 | 1.10 | 0.989 |
| | Ours, 3 seeds | 0.226 ± 0.026 | 0.302 ± 0.023 | 0.627 ± 0.074 | 1.20 | 0.924 |
| Fresh held-out, macro of 4 | Small, untouched | 0.394 | 0.612 | 0.498 | 1.34 | 0.864 |
| | Large, untouched | 0.337 | 0.541 | 0.494 | 1.31 | 0.925 |
| | Ours, 3 seeds | 0.309 ± 0.009 | 0.618 ± 0.009 | 0.468 ± 0.021 | 1.09 | 0.868 ± 0.002 |

Fresh set per source (AbsRel / delta1 / bias / delta1 scale-fixed; median true depth):

| Source | Large, untouched | Ours, 3-seed mean |
|---|---|---|
| 7-Scenes heads (0.77 m) | 0.586 / 0.057 / 1.53 / 0.932 | 0.612 / 0.140 / 1.59 / 0.882 |
| 7-Scenes stairs (1.75 m) | 0.402 / 0.272 / 1.37 / 0.853 | 0.207 / 0.626 / 1.09 / 0.821 |
| TUM freiburg2 desk (1.73 m) | 0.192 / 0.833 / 1.18 / 0.965 | 0.205 / 0.524 / 0.81 / 0.908 |
| TUM freiburg3 long office (1.69 m) | 0.168 / 0.813 / 1.14 / 0.952 | 0.211 / 0.582 / 0.88 / 0.861 |

Test set per source, 3-seed mean (AbsRel / delta1 / bias): Bonn 0.153 / 0.810 / 1.01, NYU 0.114 / 0.883 / 1.00, ARKitScenes 0.334 / 0.438 / 1.05, DIODE 0.253 / 0.571 / 1.02, TUM freiburg1 0.185 / 0.704 / 1.07, Hypersim 0.184 / 0.600 / 0.88.

What this shows:
- The frozen-test result is stable across seeds (AbsRel 0.202-0.212) and ahead of Large on every raw metric. Four of its five sources share a dataset with our training data, so this is the in-distribution result.
- Middlebury DSLR holds across seeds (0.265-0.284 vs 0.396). Close-range does not: seed 0's 0.204 was the lucky end of 0.204-0.254; the mean is 0.226, slightly behind Large's 0.211.
- On the fresh set there is no win. AbsRel is slightly lower than Large (0.309 vs 0.337), but delta1 is lower (0.468 vs 0.494), RMSE is higher (0.618 vs 0.541) and shape is clearly worse (0.868 vs 0.925, the same as untouched Small). Per source it is one clear win (stairs), one clear loss on each TUM desk/office sequence, and heads is bad for everyone.
- The TUM losses are the same failure section 14 identified: desk and office scenes full of objects at about 1.7 m are placed 12-19% too close (bias 0.81, 0.88), where Large is 14-18% too far. The tabletop data taught "objects mean near".
- 7-Scenes heads: all three models put it 53-74% too far. That is consistent enough across unrelated models to suspect the ground-truth scale or the default intrinsics of this sequence, but it was not checked.
- Shape on unfamiliar scenes is at untouched-Small level. The decoder fine-tune has improved shape only on data resembling the training sets.

Conclusion: the claim "Small + FastKAN intrinsics beats Large" holds in-distribution and on long lenses, and is not supported on fresh scenes with ordinary lenses, where the two are roughly level on scale and Large is better on shape.

Note: after the section 6 and 7 runs the ARKitScenes training pool was enlarged from 12 to 72 videos. The test set is locked to `datasets/test_set/manifest.json` and did not change; train is now 3,606 images and validation 1,056.

## 16. Low-rank adapters on the backbone

Run `runs/split5_kan_lora`: the section 14 model (split + FastKAN, lens-balanced, same data, schedule and seed 0) plus rank-8 adapters on the query and value projections of all 12 backbone blocks (147,456 parameters, learning rate 2e-4). Backbone weights stay frozen. One seed. Best validation AbsRel 0.158 at epoch 10 (section 14 seeds: about 0.18).

| Benchmark | Model | AbsRel | RMSE | delta1 | bias | delta1 scale-fixed |
|---|---|---|---|---|---|---|
| Frozen test, 5 real cameras | Large, untouched | 0.246 | - | 0.629 | 1.17 | 0.940 |
| | Section 14, 3 seeds | 0.208 ± 0.005 | - | 0.681 | 1.03 | 0.906 |
| | + adapters | 0.192 | 0.355 | 0.727 | 1.04 | 0.926 |
| Middlebury 2014 (DSLR, 23) | Large, untouched | 0.396 | 1.208 | 0.132 | 0.62 | 0.891 |
| | Section 14, 3 seeds | 0.274 ± 0.010 | 0.865 | 0.399 | 0.77 | 0.888 |
| | + adapters | 0.214 | 0.738 | 0.552 | 0.88 | 0.913 |
| Middlebury 2021 close-range (11) | Large, untouched | 0.211 | 0.283 | 0.690 | 1.10 | 0.989 |
| | Section 14, 3 seeds | 0.226 ± 0.026 | 0.302 | 0.627 | 1.20 | 0.924 |
| | + adapters | 0.337 | 0.446 | 0.425 | 1.32 | 0.932 |
| Fresh held-out (4 sources, 400) | Large, untouched | 0.337 | 0.541 | 0.494 | 1.31 | 0.925 |
| | Section 14, 3 seeds | 0.309 ± 0.009 | 0.618 | 0.468 | 1.09 | 0.868 |
| | + adapters | 0.275 | 0.592 | 0.483 | 1.04 | 0.891 |

Fresh set per source (AbsRel / delta1 / bias / delta1 scale-fixed):

| Source | Large, untouched | Section 14, 3-seed mean | + adapters |
|---|---|---|---|
| 7-Scenes heads | 0.586 / 0.057 / 1.53 / 0.932 | 0.612 / 0.140 / 1.59 / 0.882 | 0.513 / 0.152 / 1.47 / 0.905 |
| 7-Scenes stairs | 0.402 / 0.272 / 1.37 / 0.853 | 0.207 / 0.626 / 1.09 / 0.821 | 0.180 / 0.678 / 1.03 / 0.850 |
| TUM freiburg2 desk | 0.192 / 0.833 / 1.18 / 0.965 | 0.205 / 0.524 / 0.81 / 0.908 | 0.226 / 0.443 / 0.79 / 0.929 |
| TUM freiburg3 long office | 0.168 / 0.813 / 1.14 / 0.952 | 0.211 / 0.582 / 0.88 / 0.861 | 0.181 / 0.659 / 0.89 / 0.879 |

Test set per source (AbsRel / delta1 / bias): Bonn 0.129 / 0.870 / 1.05, NYU 0.107 / 0.893 / 1.00, ARKitScenes 0.336 / 0.455 / 1.08, DIODE 0.250 / 0.607 / 1.05, TUM freiburg1 0.136 / 0.810 / 1.03, Hypersim 0.209 / 0.515 / 0.87. Intrinsics-use check: focal x0.75 -> depth x0.889, focal x1.5 -> depth x1.242.

What this shows:
- Shape improved everywhere, which the frozen backbone never managed off-distribution: test 0.906 -> 0.926, Middlebury DSLR 0.888 -> 0.913, fresh 0.868 -> 0.891. It is still below Large (0.940 test, 0.925 fresh, 0.989 close-range).
- Frozen test and Middlebury DSLR are the best so far, and ahead of Large by a wide margin on raw metrics.
- Fresh set: AbsRel is now clearly below Large (0.275 vs 0.337), delta1 is level (0.483 vs 0.494), RMSE (0.592 vs 0.541) and shape remain behind. Per source it beats Large on stairs and heads and still loses on both TUM sequences.
- The "objects mean near" error did not move: TUM desk and office are still placed 11-21% too close (bias 0.79, 0.89). Adapters improved shape, not this prior.
- Middlebury close-range got clearly worse: 32% too far (0.337 vs 0.226 before and 0.211 for Large). With 11 scenes and one seed this is noisy, but it is outside the section 15 seed range (0.204-0.254).
- One seed, and the fresh set is no longer clean: it was looked at in section 15 before this design choice.

## 17. Scale from per-patch votes over several hypotheses

Run `runs/split6_kan_votes`: the section 16 model with the single global scale branch replaced by local votes. Every backbone patch (tokens of all four taps, the decoder's local log-depth, camera ray and focal lengths) feeds one shared FastKAN that outputs logits over 12 log-scale hypotheses, a confidence and a gain; the confidence-weighted mean of the logits is the image's distribution, its expectation plus a camera-only FastKAN shift is the log-scale. Extra loss: 0.2 x cross-entropy of that distribution against the true scale. 94,047 new parameters (the old branch had 110,018). Same adapters, data, schedule and seed 0. Best validation AbsRel 0.154 (section 16: 0.158).

| Benchmark | Model | AbsRel | RMSE | delta1 | bias | delta1 scale-fixed |
|---|---|---|---|---|---|---|
| Frozen test, 5 real cameras | Large, untouched | 0.246 | - | 0.629 | 1.17 | 0.940 |
| | Adapters (sec. 16) | 0.192 | 0.355 | 0.727 | 1.04 | 0.926 |
| | Adapters + votes | 0.183 | 0.353 | 0.744 | 1.03 | 0.922 |
| Middlebury 2014 (DSLR, 23) | Large, untouched | 0.396 | 1.208 | 0.132 | 0.62 | 0.891 |
| | Adapters (sec. 16) | 0.214 | 0.738 | 0.552 | 0.88 | 0.913 |
| | Adapters + votes | 0.257 | 0.872 | 0.480 | 0.91 | 0.923 |
| Middlebury 2021 close-range (11) | Large, untouched | 0.211 | 0.283 | 0.690 | 1.10 | 0.989 |
| | Adapters (sec. 16) | 0.337 | 0.446 | 0.425 | 1.32 | 0.932 |
| | Adapters + votes | 0.237 | 0.327 | 0.613 | 1.19 | 0.894 |
| Fresh held-out (4 sources, 400) | Large, untouched | 0.337 | 0.541 | 0.494 | 1.31 | 0.925 |
| | Adapters (sec. 16) | 0.275 | 0.592 | 0.483 | 1.04 | 0.891 |
| | Adapters + votes | 0.267 | 0.585 | 0.481 | 1.04 | 0.886 |

Fresh set per source, adapters + votes (AbsRel / delta1 / bias / delta1 scale-fixed): heads 0.439 / 0.242 / 1.40 / 0.912, stairs 0.204 / 0.644 / 1.10 / 0.835, TUM freiburg2 desk 0.227 / 0.450 / 0.79 / 0.924, TUM freiburg3 office 0.197 / 0.587 / 0.86 / 0.874.

Test set per source (AbsRel / delta1 / bias): Bonn 0.123 / 0.867 / 1.05, NYU 0.101 / 0.901 / 0.99, ARKitScenes 0.323 / 0.468 / 1.08, DIODE 0.241 / 0.638 / 1.04, TUM freiburg1 0.128 / 0.847 / 1.01, Hypersim 0.161 / 0.716 / 1.01.

What this shows:
- Best frozen-test result so far (0.183) and close-range recovered from 0.337 to 0.237, back inside the section 15 seed range.
- Middlebury DSLR got worse (0.214 -> 0.257) and the fresh set is unchanged within noise (0.275 -> 0.267).
- The target failure did not move: TUM desk and office are still 14-21% too close (bias 0.79, 0.86).
- One seed each for sections 16 and 17; the differences between them are within what seeds produced in section 15, except close-range.

### Oracle check: how much can a per-image correction buy? (`diagnostics/oracle_correction.py`, fresh set)

Each image's prediction is fitted to its ground truth three ways; AbsRel / delta1 after the fit, all 400 images:

| Model | raw | scale only | scale + gain (log-affine) | scale + shift in metres | median fitted shift |
|---|---|---|---|---|---|
| Small, untouched | 0.394 / 0.498 | 0.105 / 0.864 | 0.103 / 0.889 | 0.102 / 0.874 | -0.03 m |
| Large, untouched | 0.337 / 0.494 | 0.080 / 0.925 | 0.074 / 0.937 | 0.084 / 0.917 | +0.19 m |
| Ours, adapters | 0.275 / 0.483 | 0.097 / 0.891 | 0.093 / 0.917 | 0.091 / 0.904 | -0.13 m |
| Ours, adapters + votes | 0.267 / 0.481 | 0.100 / 0.886 | 0.095 / 0.911 | 0.094 / 0.900 | -0.04 m |

- Almost all of the error is the single per-image scale: fixing it alone takes our model from 0.267 to 0.100.
- A perfect metre shift on top of a perfect scale buys 0.006 AbsRel and about 0.014 delta1. A perfect gain (which the branch already outputs) buys about the same AbsRel and more delta1 (0.911 vs 0.900).
- The fitted shifts are small (median -0.04 to -0.13 m for ours) and change sign between sources.
- So the metre shift is not where the fresh-set error is. Section 18 trains it anyway as a single-change run.

## 18. Metre shift, and which classical scale cue would fix the hard scenes

### Metre shift (`runs/split7_kan_shift`)

Section 17 model plus a per-image shift in metres voted by the patches (bounded to +-1 m): depth = exp((1 + gain) * shape + scale) + shift. One seed. Best validation AbsRel 0.157 (section 17: 0.154).

| Benchmark | Adapters + votes (sec. 17) | + metre shift |
|---|---|---|
| Frozen test (AbsRel / delta1 / bias / shape) | 0.183 / 0.744 / 1.03 / 0.922 | 0.183 / 0.726 / 1.01 / 0.922 |
| Middlebury 2014 | 0.257 / 0.480 / 0.91 / 0.923 | 0.252 / 0.492 / 0.89 / 0.927 |
| Middlebury 2021 close-range | 0.237 / 0.613 / 1.19 / 0.894 | 0.373 / 0.431 / 1.36 / 0.930 |
| Fresh held-out | 0.267 / 0.481 / 1.04 / 0.886 | 0.277 / 0.459 / 1.03 / 0.890 |

No gain anywhere, as the oracle check in section 17 predicted; close-range is worse. TUM desk / office bias 0.77 / 0.86, unchanged. The metre shift is dropped.

### Ground plane and known-object cues (`diagnostics/geometry_cues.py`, the two fresh TUM sequences, 200 images)

The motion-capture pose gives the true gravity direction (roll included) and camera height (mean 1.56 m). Each prediction is rescaled per image and scored (AbsRel / delta1 / bias). "ground-auto": lowest 2% of the predicted cloud is taken as floor and matched to the known height. "ground-mask": true floor pixels, plane equation. "anchor": one 10% x 10% patch has known depth (stand-in for a known-size object). Floor counts as visible at >= 5% of the image.

| Model, subset | raw | best single scale | ground-auto | ground-mask | anchor |
|---|---|---|---|---|---|
| Ours, fr2 desk, floor visible (64 of 100) | 0.207 / 0.520 / 0.80 | 0.074 / 0.940 | 0.086 / 0.887 / 0.95 | 0.081 / 0.907 / 0.97 | 0.111 / 0.858 / 1.03 |
| Ours, fr2 desk, all 100 | 0.227 / 0.450 / 0.79 | 0.081 / 0.924 | 0.113 / 0.820 / 0.98 | - | 0.125 / 0.826 / 1.04 |
| Ours, fr3 office, all 100 (floor visible in 4) | 0.197 / 0.587 / 0.86 | 0.100 / 0.874 | 0.180 / 0.772 / 1.05 | - | 0.177 / 0.769 / 1.06 |
| Ours, both, all 200 | 0.212 / 0.519 / 0.82 | 0.091 / 0.899 | 0.147 / 0.796 / 1.02 | - | 0.151 / 0.797 / 1.05 |
| Large, both, floor visible (68) | 0.184 / 0.854 / 1.17 | 0.052 / 0.964 | 0.058 / 0.961 / 0.99 | 0.063 / 0.962 / 1.02 | 0.072 / 0.956 / 1.02 |
| Large, both, all 200 | 0.180 / 0.823 / 1.16 | 0.055 / 0.958 | 0.119 / 0.884 / 1.08 | - | 0.074 / 0.938 / 1.01 |

- With the floor in view and the camera height known, the ground plane removes almost all of the scale error (ours 0.207 -> 0.086, ceiling 0.074), without needing a floor mask.
- The floor is in view in 64% of the desk images and 4% of the office images. Without it the cue does little (office 0.197 -> 0.180).
- A known-depth patch helps less for our model (0.212 -> 0.151) than for Large (0.180 -> 0.074), because our relative depth is less consistent across the image: a local anchor only fixes the whole image if shape is right.
- Both cues need information a single image with intrinsics does not contain (camera height, object size). Depth from defocus was not testable: fixed-focus sensors, no aperture or focus metadata, 252x336 inputs.
- Our model's error on these scenes is equivalent to assuming the camera is about 1.2 m above the floor when it is about 1.5 m.

## 19. Would a predicted camera height be easier than a predicted scale? (`diagnostics/support_plane_stats.py`)

Planes fitted to ground-truth depth (sequential RANSAC). Support plane = lowest plane below the camera, normal within 60 degrees of image-down, at least 5% of the image. No gravity metadata is used, so floors and tabletops are not told apart. "Spread" is the standard deviation of the log.

Check against motion capture on the two fresh TUM sequences: where the floor is truly visible (68 of 200) the fit is accurate (fitted / true height median 0.993, p10 0.916; normal 2.5 degrees off gravity). But a support plane is fitted in 197 of 200 images, and 130 of those are a desk or other surface, not the floor.

| Set, source | Images | With support plane | Height above it p10 / p50 / p90 (m) | Spread of height | Spread of median depth |
|---|---|---|---|---|---|
| Train, Bonn | 1929 | 67% | 0.90 / 1.43 / 1.90 | 0.48 | 0.28 |
| Train, NYU | 491 | 83% | 0.68 / 1.36 / 1.92 | 0.40 | 0.28 |
| Train, ARKitScenes | 995 | 33% | 0.47 / 1.03 / 1.48 | 0.53 | 0.30 |
| Train, DIODE | 175 | 35% | 0.67 / 1.83 / 5.47 | 1.12 | 0.66 |
| Train, SUN Kinect v2 | 819 | 91% | 1.14 / 1.37 / 1.65 | 0.19 | 0.23 |
| Train, SUN RealSense | 779 | 89% | 0.70 / 1.29 / 1.44 | 0.38 | 0.19 |
| Train, SUN Xtion | 774 | 75% | 0.75 / 1.36 / 1.65 | 0.40 | 0.36 |
| Train, BOP long lens | 1250 | 88% | 0.23 / 0.44 / 0.63 | 0.43 | 0.34 |
| Train, BOP standard | 726 | 74% | 0.32 / 0.62 / 1.32 | 0.67 | 0.59 |
| Train, all | 7954 | 72% | 0.39 / 1.25 / 1.69 | 0.64 | 0.48 |
| Frozen test, all | 440 | 64% | 0.55 / 1.27 / 1.88 | 0.60 | 0.50 |
| Fresh held-out, all | 400 | 94% | 0.49 / 1.29 / 1.66 | 0.47 | 0.40 |

- Supervision would be available: 72% of training images have a support plane.
- The distance from the camera to that plane varies more than the scene's median depth in every source except SUN Kinect v2 (tripod, floor in view), where the two are about equal. So "camera height is a narrower target than depth" does not hold on this data.
- Without gravity, most fitted planes in desk scenes are the desk, so the quantity is "distance to the nearest support surface", which is close to the scene scale itself.
- Conclusion: routing scale through a predicted plane height replaces one guess by an equally hard one. Not built.

## 20. Larger adapters

Run `runs/split8_kan_lora16`: the section 17 model (adapters + votes) with rank-16 adapters on query, key, value and output of all 12 backbone blocks (589,824 parameters, was 147,456 at rank 8 on query and value). Same data, schedule, seed 0. Best validation AbsRel 0.153 (section 17: 0.154).

| Benchmark (AbsRel / delta1 / bias / delta1 scale-fixed) | Large, untouched | Adapters r8 + votes (sec. 17) | Adapters r16 all + votes |
|---|---|---|---|
| Frozen test, 5 real cameras | 0.246 / 0.629 / 1.17 / 0.940 | 0.183 / 0.744 / 1.03 / 0.922 | 0.184 / 0.735 / 1.03 / 0.927 |
| Middlebury 2014 | 0.396 / 0.132 / 0.62 / 0.891 | 0.257 / 0.480 / 0.91 / 0.923 | 0.259 / 0.410 / 0.83 / 0.943 |
| Middlebury 2021 close-range | 0.211 / 0.690 / 1.10 / 0.989 | 0.237 / 0.613 / 1.19 / 0.894 | 0.311 / 0.447 / 1.27 / 0.929 |
| Fresh held-out | 0.337 / 0.494 / 1.31 / 0.925 | 0.267 / 0.481 / 1.04 / 0.886 | 0.269 / 0.523 / 1.06 / 0.877 |

Fresh set per source (AbsRel / delta1 / bias / shape): heads 0.521 / 0.182 / 1.46 / 0.870, stairs 0.178 / 0.681 / 1.05 / 0.843, TUM freiburg2 desk 0.190 / 0.573 / 0.83 / 0.923, TUM freiburg3 office 0.186 / 0.657 / 0.89 / 0.872.

- Four times the adapter capacity changed nothing that matters: frozen test and fresh set are the same within noise, and shape on the fresh set did not rise (0.877 vs 0.886).
- Across the four adapter runs (sections 16-18, 20) the results sit in a narrow band: frozen test 0.183-0.192, fresh 0.267-0.277, fresh shape 0.877-0.891. Close-range moves between 0.24 and 0.37 from run to run (11 scenes), so it cannot rank these variants.
- The model has plateaued for this training set (7,954 images). Capacity of the adapters and the form of the scale branch are not the limit.

## 21. Is scale predictable from the existing features with a fixed shape and reference? (`diagnostics/scale_fit.py`)

Prompted by an external review of the workspace, which pointed out that the scale branch learns relative to a decoder level the shape loss leaves free to drift. Test: the section 17 model is frozen completely. Per image, target log-scale = median over valid pixels of (log true depth - frozen shape), gain fixed at zero. FastKAN scale heads are trained on cached pooled backbone features (class + mean patch token of the four taps) and the camera, one equal-weight L1 term per image, validation picks the epoch, 3 seeds. Training views: two lens-balanced augmented passes (15,908).

Mean |log-scale error| (0.10 is about 10% off):

| Predictor | Training views | Validation (unseen scenes, training cameras) | Frozen test | Fresh set | Middlebury (34) |
|---|---|---|---|---|---|
| Training median (constant) | 0.375 | 0.385 | 0.379 | 0.320 | 0.417 |
| Current model's own scale | 0.103 | 0.093 | 0.142 | 0.226 | 0.244 |
| Head: camera only | 0.329 | 0.289 | 0.399 | 0.369 | 0.587 |
| Head: absolute | 0.062 | 0.094 | 0.142 | 0.201 | 0.286 |
| Head: absolute + decoder level as input | 0.054 | 0.092 | 0.141 | 0.204 | 0.270 |
| Head: residual on decoder level (current formulation) | 0.064 | 0.090 | 0.138 | 0.215 | 0.255 |

Signed error on the hard sources (negative = too close), current model / absolute head / residual head: TUM freiburg2 desk -0.24 / -0.25 / -0.26, TUM freiburg3 office -0.16 / -0.12 / -0.15, 7-Scenes heads +0.33 / +0.29 / +0.31, Middlebury 2014 -0.15 / -0.21 / -0.22, Middlebury 2021 close +0.17 / +0.19 / +0.16.

- With shape and reference fixed, the heads reach the same held-out error as the jointly trained model (validation 0.090-0.094 vs 0.093, test 0.138-0.142 vs 0.142). The drifting reference is not what limits scale.
- Residual and absolute formulations are indistinguishable.
- Training scale is fitted well (0.054-0.064) and the error grows with distance from the training data: 0.09 on new scenes of the same cameras, 0.14 on the test set, 0.20-0.29 on unseen datasets. The per-source signs are the same for every head.
- Camera alone is barely better than a constant; the scale is read from image content, and that reading does not carry over to unfamiliar scenes.
- By the review's own criterion (training scale accurate, unseen-scene scale fails) this is a transferable-cue / data-coverage problem. Caveats: pooled features only, features from a backbone adapted on the same data, and the fresh set and Middlebury have been used repeatedly.

Confirmed from the review and still to fix: the source-balanced sampler gives the 16 Hypersim training images about 10% of all draws, and the metric loss is averaged over valid pixels of the whole batch, so heavily padded images weigh less.

## 22. More rooms (ARKitScenes Training fold) with corrected sampling and loss weighting

Run `runs/split9_kan_data`: the section 17 model (rank-8 adapters + votes, seed 0) with three changes at once. (1) 300 more ARKitScenes videos, one per visit, from the Training fold as a train/val-only source `arkit_train` (4,006 training images; training set 7,954 -> 11,960; frozen test set unchanged, 440 images). (2) Sources drawn with probability proportional to the square root of their size (was equal per source, which gave the 16 Hypersim images about 10% of draws). (3) Losses averaged per image (was per valid pixel of the batch). 12 epochs of 5,000 samples. Validation is not comparable with earlier runs (one more source).

| Benchmark (AbsRel / delta1 / bias / delta1 scale-fixed) | Large, untouched | Section 17 | This run |
|---|---|---|---|
| Frozen test, 5 real cameras | 0.246 / 0.629 / 1.17 / 0.940 | 0.183 / 0.744 / 1.03 / 0.922 | 0.172 / 0.753 / 1.02 / 0.929 |
| Middlebury 2014 | 0.396 / 0.132 / 0.62 / 0.891 | 0.257 / 0.480 / 0.91 / 0.923 | 0.266 / 0.385 / 0.82 / 0.929 |
| Middlebury 2021 close-range | 0.211 / 0.690 / 1.10 / 0.989 | 0.237 / 0.613 / 1.19 / 0.894 | 0.246 / 0.635 / 1.23 / 0.939 |
| Fresh held-out | 0.337 / 0.494 / 1.31 / 0.925 | 0.267 / 0.481 / 1.04 / 0.886 | 0.262 / 0.494 / 1.04 / 0.892 |

Fresh held-out RMSE: Large 0.541, section 17 0.585, this run 0.576.

Test per source (AbsRel / delta1 / bias): Bonn 0.124 / 0.884 / 1.06, NYU 0.110 / 0.882 / 0.99, ARKitScenes 0.246 / 0.594 / 1.06 (was 0.323 / 0.468), DIODE 0.249 / 0.570 / 0.96, TUM freiburg1 0.131 / 0.833 / 1.00, Hypersim 0.163 / 0.749 / 1.04.

Fresh per source (AbsRel / delta1 / bias / shape): heads 0.458 / 0.221 / 1.42 / 0.910, stairs 0.163 / 0.748 / 1.09 / 0.855, TUM freiburg2 desk 0.221 / 0.458 / 0.79 / 0.929, TUM freiburg3 office 0.205 / 0.549 / 0.85 / 0.875.

- Best frozen-test result so far (0.172). The gain is where the new data is: ARKitScenes test 0.323 -> 0.246; other test sources unchanged.
- Unseen datasets barely moved: fresh 0.267 -> 0.262, Middlebury unchanged within noise. TUM desk and office are still 15-21% too close.
- More rooms from a dataset already in the mix improve that dataset, not scenes of a different kind.

### When did the too-close error on TUM desk / office appear? (older checkpoints on the fresh set)

| Model | freiburg2 desk bias | freiburg3 office bias | Fresh AbsRel | Fresh shape |
|---|---|---|---|---|
| Small, untouched | 1.09 | 1.13 | 0.394 | 0.864 |
| FiLM + FastKAN, original data (sec. 8) | 0.81 | 0.93 | 0.281 | 0.854 |
| Split, + SUN RGB-D, before tabletop data (sec. 12) | 0.87 | 0.95 | 0.312 | 0.867 |
| Split, + tabletop data (sec. 13) | 0.81 | 0.89 | 0.341 | 0.868 |
| Adapters + votes (sec. 17) | 0.79 | 0.86 | 0.267 | 0.886 |
| This run | 0.79 | 0.85 | 0.262 | 0.892 |

- The error was already there in the first fine-tuned model, before any tabletop data (desk 0.81). Tabletop data made the office sequence worse (0.95 -> 0.89) but did not create the problem.
- Every fine-tuned variant places these scenes 5-21% too close, whatever the data mix or architecture.
