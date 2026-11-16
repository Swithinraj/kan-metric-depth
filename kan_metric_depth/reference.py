"""Run an unmodified Depth Anything V2 checkpoint through its own Hugging Face image processor."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

RELATIVE_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"


def load_depth_anything(device: torch.device, model_id: str = RELATIVE_SMALL):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.inference_mode()
def predict(processor, model, rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """rgb: B,3,H,W in [0,1]. Returns the model output (relative disparity, or metres for metric checkpoints) at H,W."""
    height, width = rgb.shape[-2:]
    images = [Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for frame in rgb.cpu()]
    inputs = processor(images=images, return_tensors="pt").to(device)
    output = model(**inputs).predicted_depth
    return F.interpolate(output.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False).squeeze(1)
