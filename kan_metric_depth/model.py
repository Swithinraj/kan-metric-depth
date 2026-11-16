"""Depth Anything V2 (metric) conditioned on camera intrinsics through FastKAN (or MLP, or nothing).

The intrinsics enter as a per-location camera code: the viewing ray of that location
((u - cx) / fx, (v - cy) / fy) plus the log focal lengths relative to image size. A
conditioner turns the code into FiLM (scale, shift) for
  A. the four backbone token taps,
  B. the four DPT fusion outputs,
and C. a global log-scale added to the predicted log-depth.
Every conditioner ends in a zero-initialised layer, so the model starts as the pretrained one.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
CODE_DIM = 4


class FastKANLayer(nn.Module):
    """KAN layer with Gaussian radial bases on every input (FastKAN), applied per location as a 1x1 conv."""

    def __init__(self, in_dim: int, out_dim: int, grid: int = 8, low: float = -2.0, high: float = 2.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(low, high, grid).view(1, 1, grid, 1, 1))
        self.width = (high - low) / (grid - 1)
        self.spline = nn.Conv2d(in_dim * grid, out_dim, 1)
        self.base = nn.Conv2d(in_dim, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-(((x.float().unsqueeze(2) - self.centers) / self.width) ** 2))
        return self.spline(basis.flatten(1, 2)) + self.base(F.silu(x.float()))


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update (starts at zero, so the layer starts unchanged)."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.randn(rank, base.in_features) / base.in_features ** 0.5)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype))


def conditioner(kind: str, in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """in_dim -> hidden -> out_dim, last layer zero-initialised. The MLP is widened to match the KAN's size."""
    if kind == "kan":
        last = FastKANLayer(hidden, out_dim)
        net = nn.Sequential(FastKANLayer(in_dim, hidden), nn.GroupNorm(1, hidden), last)
        zero = [last.spline, last.base]
    elif kind == "mlp":
        kan_params = (in_dim * 8 + in_dim) * hidden + hidden + 2 * hidden + (hidden * 8 + hidden) * out_dim + out_dim
        width = max(hidden, round((kan_params - out_dim) / (in_dim + 1 + 2 + out_dim)))
        last = nn.Conv2d(width, out_dim, 1)
        net = nn.Sequential(nn.Conv2d(in_dim, width, 1), nn.GroupNorm(1, width), nn.SiLU(), last)
        zero = [last]
    else:
        raise ValueError(kind)
    for module in zero:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return net


def camera_code(intrinsics: torch.Tensor, image_size: tuple[int, int], grid_size: tuple[int, int]) -> torch.Tensor:
    """[B,4] intrinsics (fx, fy, cx, cy) in pixels of image_size -> [B,4,h,w] code at grid_size (resolution independent)."""
    height, width = image_size
    h, w = grid_size
    fx, fy, cx, cy = (intrinsics[:, i].view(-1, 1, 1) for i in range(4))
    u = ((torch.arange(w, device=intrinsics.device) + 0.5) * width / w).view(1, 1, w)
    v = ((torch.arange(h, device=intrinsics.device) + 0.5) * height / h).view(1, h, 1)
    ray_x = ((u - cx) / fx).expand(-1, h, w)
    ray_y = ((v - cy) / fy).expand(-1, h, w)
    log_fx = torch.log(fx / width).expand(-1, h, w)
    log_fy = torch.log(fy / height).expand(-1, h, w)
    return torch.stack((ray_x, ray_y, 2 * log_fx, 2 * log_fy), dim=1)


class IntrinsicsDepth(nn.Module):
    """cond: 'none' (plain fine-tune control), 'mlp' or 'kan'."""

    def __init__(self, cond: str = "kan", work_size: tuple[int, int] = (392, 518), hidden: int = 16,
                 model_id: str = MODEL_ID):
        super().__init__()
        if cond not in ("none", "mlp", "kan"):
            raise ValueError("cond must be none, mlp or kan")
        if work_size[0] % 14 or work_size[1] % 14:
            raise ValueError("work_size must be multiples of 14")
        self.cond, self.work_size = cond, work_size
        self.da = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.da.backbone.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        config = self.da.config
        token_dim, fusion_dim = config.reassemble_hidden_size, config.fusion_hidden_size
        if cond != "none":
            self.tap_film = nn.ModuleList(conditioner(cond, CODE_DIM, hidden, 2 * token_dim) for _ in range(4))
            self.fusion_film = nn.ModuleList(conditioner(cond, CODE_DIM, hidden, 2 * fusion_dim) for _ in range(4))
            self.pool_proj = nn.Linear(fusion_dim, 6)
            self.scale = conditioner(cond, 6 + 2, hidden, 1)

    def trainable_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained = list(self.da.neck.parameters()) + list(self.da.head.parameters())
        ids = {id(p) for p in pretrained}
        new = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return pretrained, new

    def train(self, mode: bool = True):
        super().train(mode)
        self.da.backbone.eval()
        return self

    @staticmethod
    def _film(features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gamma, beta = params.to(features.dtype).chunk(2, dim=1)
        return features * (1 + gamma) + beta

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """rgb [B,3,H,W] in 0..1, intrinsics [B,4] in pixels of (H,W). Returns log-depth [B,1,H,W] (meters)."""
        return self._decode(self._taps(rgb), intrinsics, rgb.shape[-2:])

    def _taps(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        work = F.interpolate(rgb, size=self.work_size, mode="bilinear", align_corners=False)
        work = (work - self.mean) / self.std
        with torch.set_grad_enabled(self.training and bool(getattr(self, "lora_rank", 0))):
            return list(self.da.backbone.forward_with_filtered_kwargs(work).feature_maps)

    def _decode(self, taps: list[torch.Tensor], intrinsics: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        ph, pw = self.work_size[0] // 14, self.work_size[1] // 14
        conditioned = self.cond != "none"
        if conditioned:
            code = camera_code(intrinsics, size, (ph, pw))
            for i, tap in enumerate(taps):
                params = self.tap_film[i](code).flatten(2).transpose(1, 2)
                taps[i] = torch.cat((tap[:, :1], self._film_tokens(tap[:, 1:], params)), dim=1)
        neck = self.da.neck
        features = [neck.convs[i](f) for i, f in enumerate(neck.reassemble_stage(taps, ph, pw))][::-1]
        fused = None
        for i, layer in enumerate(neck.fusion_stage.layers):
            target = features[i + 1].shape[2:] if i + 1 < len(features) else None
            fused = layer(features[i], size=target) if fused is None else layer(fused, features[i], size=target)
            if conditioned:
                fused = self._film(fused, self.fusion_film[i](camera_code(intrinsics, size, fused.shape[2:])))
        head = self.da.head
        x = head.conv1(fused)
        x = F.interpolate(x, self.work_size, mode="bilinear", align_corners=True)
        logit = head.conv3(head.activation1(head.conv2(x))).float()
        log_depth = F.logsigmoid(logit) + torch.log(torch.tensor(float(head.max_depth)))
        if conditioned:
            pooled = torch.tanh(self.pool_proj(fused.float().mean(dim=(2, 3))))
            global_code = torch.cat((pooled, camera_code(intrinsics, size, (1, 1))[:, 2:, 0, 0]), dim=1)
            log_depth = log_depth + self.scale(global_code[:, :, None, None])
        return F.interpolate(log_depth, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _film_tokens(tokens: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gamma, beta = params.to(tokens.dtype).chunk(2, dim=2)
        return tokens * (1 + gamma) + beta


class SplitScaleDepth(IntrinsicsDepth):
    """Shape and scale split apart.

    The decoder gives log-depth; its per-image mean is removed, leaving shape only. One scale branch
    (FastKAN or MLP) reads the camera (focal lengths, principal point) together with image content
    (class token and mean patch token of all four backbone taps) and outputs the image's log-scale
    and a gain on the shape:  log depth = (1 + gain) * shape + log-scale.
    The branch starts at zero, so the model starts as the pretrained one. `use_camera=False` hides
    the intrinsics from the branch (content-only control).

    `votes=K` replaces the single global branch by local votes: every backbone patch (its tokens from all
    four taps, the decoder's local log-depth, its camera ray and the focal lengths) gives logits over K
    log-scale hypotheses, a confidence and a gain. The confidence-weighted mean of the logits is the image's
    distribution over hypotheses; its expectation, plus a camera-only shift, is the log-scale.
    """

    CONTENT_DIM = 32
    VOTE_DIM = 48

    def __init__(self, cond: str = "kan", work_size: tuple[int, int] = (392, 518), hidden: int = 32,
                 model_id: str = MODEL_ID, use_camera: bool = True, lora_rank: int = 0, votes: int = 0, metre_shift: bool = False, lora_all: bool = False):
        if cond not in ("mlp", "kan"):
            raise ValueError("cond must be mlp or kan")
        super().__init__("none", work_size, model_id=model_id)
        self.branch_kind, self.use_camera = cond, use_camera
        token_dim = self.da.config.reassemble_hidden_size
        self.votes, self.metre_shift = votes, metre_shift and bool(votes)
        if votes:
            self.vote_proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(4 * token_dim, self.VOTE_DIM), nn.LayerNorm(self.VOTE_DIM))
            self.vote = conditioner(cond, self.VOTE_DIM + 1 + CODE_DIM, hidden, votes + 2 + int(metre_shift))
            self.shift = conditioner(cond, 4, 16, 1)
            self.register_buffer("centres", torch.linspace(-1.5, 1.5, votes))
        else:
            self.content = nn.Sequential(nn.Dropout(0.1), nn.Linear(8 * token_dim, self.CONTENT_DIM), nn.LayerNorm(self.CONTENT_DIM))
            self.branch = conditioner(cond, self.CONTENT_DIM + 2 + (4 if use_camera else 0), hidden, 2)
        # Optional low-rank adapters on the query and value projections of every backbone block (backbone weights stay frozen).
        self.lora_rank = lora_rank
        if lora_rank:
            for block in self.da.backbone.encoder.layer:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj") if lora_all else ("q_proj", "v_proj"):
                    setattr(block.attention, name, LoRALinear(getattr(block.attention, name), lora_rank))

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.da.backbone.named_parameters() if p.requires_grad]

    def trainable_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        pretrained, new = super().trainable_groups()
        lora = {id(p) for p in self.lora_parameters()}
        return pretrained, [p for p in new if id(p) not in lora]

    def parts(self, rgb: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns shape [B,1,H,W] (zero-mean log-depth), log-scale [B,1,1,1], gain [B,1,1,1]."""
        height, width = rgb.shape[-2:]
        taps = self._taps(rgb)
        raw = self._decode(taps, intrinsics, (height, width)).float()
        level = raw.mean(dim=(2, 3), keepdim=True)
        shape = raw - level
        if self.votes:
            return self._vote_parts(taps, raw, level, shape, intrinsics)
        tokens = torch.cat([torch.cat((t[:, 0], t[:, 1:].mean(dim=1)), dim=1) for t in taps], dim=1).float()
        level_in = level.detach().flatten(1)
        code = [self.content(tokens), level_in - 1.0, shape.detach().flatten(1).std(dim=1, keepdim=True)]
        if self.use_camera:
            fx, fy, cx, cy = intrinsics.float().unbind(dim=1)
            code.append(torch.stack((2 * torch.log(fx / width), 2 * torch.log(fy / height),
                                     2 * (cx / width - 0.5), 2 * (cy / height - 0.5)), dim=1))
        out = self.branch(torch.cat(code, dim=1)[:, :, None, None])
        # The decoder's own level is only a starting point (detached): metric gradients reach the branch alone.
        return shape, level.detach() + out[:, :1], out[:, 1:]

    def _vote_parts(self, taps, raw, level, shape, intrinsics):
        size = raw.shape[-2:]
        ph, pw = self.work_size[0] // 14, self.work_size[1] // 14
        tokens = self.vote_proj(torch.cat([t[:, 1:] for t in taps], dim=2).float())
        feat = tokens.transpose(1, 2).reshape(raw.shape[0], -1, ph, pw)
        local = F.interpolate(raw.detach(), (ph, pw), mode="area") - 1.0
        out = self.vote(torch.cat((feat, local, camera_code(intrinsics.float(), size, (ph, pw))), dim=1)).float().flatten(2)
        weight = out[:, self.votes].softmax(dim=1)[:, None]  # [B,1,N] confidence over patches
        logits = (out[:, :self.votes] * weight).sum(dim=2)    # [B,K] pooled hypotheses
        gain = (out[:, self.votes + 1:self.votes + 2] * weight).sum(dim=2)
        # Optional per-image shift in metres (bounded to +-1 m), added after the log-space scale and gain.
        self.shift_m = torch.tanh((out[:, self.votes + 2:] * weight).sum(dim=2))[:, :, None, None] if self.metre_shift else None
        fx, fy, cx, cy = intrinsics.float().unbind(dim=1)
        camera = torch.stack((2 * torch.log(fx / size[1]), 2 * torch.log(fy / size[0]),
                              2 * (cx / size[1] - 0.5), 2 * (cy / size[0] - 0.5)), dim=1)
        base = level.detach().flatten(1) + self.shift(camera[:, :, None, None]).float().flatten(1)
        delta = (logits.softmax(dim=1) * self.centres).sum(dim=1, keepdim=True)
        self.aux = (logits.log_softmax(dim=1), base)
        return shape, (base + delta)[:, :, None, None], gain[:, :, None, None]

    def vote_loss(self, shape: torch.Tensor, gain: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of the pooled hypothesis distribution against the image's true log-scale (soft two-bin target)."""
        log_prob, base = self.aux
        resid = depth.clamp_min(1e-3).log() - ((1 + gain) * shape).detach().float()
        target = torch.stack([r[v].median() if v.any() else r.new_zeros(()) for r, v in zip(resid, valid)]) - base.detach()[:, 0]
        step = self.centres[1] - self.centres[0]
        pos = ((target - self.centres[0]) / step).clamp(0, self.votes - 1 - 1e-4)
        low, frac = pos.floor().long(), pos - pos.floor()
        dist = torch.zeros_like(log_prob).scatter_(1, low[:, None], (1 - frac)[:, None]).scatter_add_(1, low[:, None] + 1, frac[:, None])
        keep = torch.stack([v.any() for v in valid]).float()
        return -((dist * log_prob).sum(dim=1) * keep).sum() / keep.sum().clamp_min(1)

    def compose(self, shape: torch.Tensor, scale: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """Log-depth from the parts of the latest `parts` call: exp((1 + gain) * shape + scale) + shift in metres."""
        log_depth = (1 + gain) * shape + scale
        if getattr(self, "metre_shift", False):
            log_depth = (log_depth.float().exp() + self.shift_m).clamp_min(1e-2).log()
        return log_depth

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        return self.compose(*self.parts(rgb, intrinsics))
