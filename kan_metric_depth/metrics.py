"""Scores for one image: raw metric error plus the same error after a per-image median-scale fix (shape only)."""

import torch

KEYS = ("AbsRel", "RMSE", "delta1", "bias", "median_AbsRel", "median_delta1")


def score(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """pred, gt: valid pixels in metres. bias = median(pred / gt): 1.0 is the right scale, above 1 is too far."""
    p, t = pred.double(), gt.double()
    out = {"AbsRel": ((p - t).abs() / t).mean().item(), "RMSE": ((p - t) ** 2).mean().sqrt().item(),
           "delta1": (torch.maximum(p / t, t / p) < 1.25).double().mean().item(),
           "bias": (p / t).median().item()}
    fixed = p * (t / p).median()
    out["median_AbsRel"] = ((fixed - t).abs() / t).mean().item()
    out["median_delta1"] = (torch.maximum(fixed / t, t / fixed) < 1.25).double().mean().item()
    return out
