"""Paper-specified Q/V AtrousLoRA paths for SAM's BHWC attention input.

The paper does not specify convolution branch widths, convolution biases, or
initialization. Here each branch defaults to ``rank`` channels, convolutions use
biases, and the usual zero-initialized LoRA up-projection preserves the pretrained
function at initialization. These reconstruction choices are configurable or
documented; they are not recovered original implementation details.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class AtrousAttentionModule(nn.Module):
    """Four dilated 3x3 branches plus an unprojected global-pooling branch.

    Implements manuscript Algorithm 1: concatenate, 1x1 projection, BN, ReLU,
    and a rank-to-one sigmoid spatial gate. No branch-specific nonlinearities,
    pooling projection, residual identity, or extra attention are introduced.
    """

    def __init__(
        self,
        rank: int = 4,
        dilations: Sequence[int] = (1, 6, 12, 18),
        branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if len(dilations) != 4 or any(int(d) != d or d <= 0 for d in dilations):
            raise ValueError("AAM requires exactly four positive integer dilations")
        branch_channels = rank if branch_channels is None else branch_channels
        if branch_channels <= 0:
            raise ValueError("branch_channels must be positive")
        self.rank = rank
        self.dilations = tuple(int(d) for d in dilations)
        self.branch_channels = branch_channels
        self.branches = nn.ModuleList(
            nn.Conv2d(
                rank, branch_channels, kernel_size=3, padding=d, dilation=d,
                bias=True,
            )
            for d in self.dilations
        )
        self.project = nn.Conv2d(4 * branch_channels + rank, rank, 1, bias=True)
        self.norm = nn.BatchNorm2d(rank)
        self.activation = nn.ReLU(inplace=False)
        self.gate = nn.Conv2d(rank, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.rank:
            raise ValueError(f"AAM expects [B,{self.rank},H,W], got {tuple(x.shape)}")
        # expand is mathematically identical to broadcasting the pooled feature;
        # unlike an additional convolution it introduces no extra parameters.
        global_context = x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
        features = torch.cat([branch(x) for branch in self.branches] + [global_context], dim=1)
        features = self.activation(self.norm(self.project(features)))
        return features * self.gate(features).sigmoid()


class LowRankPath(nn.Module):
    """An independent channel-linear down/(AAM)/up path, with alpha/r scaling."""

    def __init__(
        self,
        dim: int,
        rank: int = 4,
        alpha: float = 4.0,
        atrous: bool = True,
        dilations: Sequence[int] = (1, 6, 12, 18),
        branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.down = nn.Linear(dim, rank, bias=False)
        self.aam = (
            AtrousAttentionModule(rank, dilations, branch_channels) if atrous else None
        )
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("SAM QKV adapters expect a BHWC grid, including window batches")
        low_rank = self.down(x)
        if self.aam is not None:
            low_rank = self.aam(low_rank.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.up(low_rank) * self.scale


class QKVWithAdapters(nn.Module):
    """Wrap a frozen fused QKV layer; independently update Q and V, preserve K."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        atrous: bool = True,
        dilations: Sequence[int] = (1, 6, 12, 18),
        branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features:
            raise ValueError("Expected SAM's fused Linear(dim, 3*dim) QKV projection")
        self.base = base
        self.base.requires_grad_(False)
        self.dim = base.in_features
        kwargs = {
            "dim": self.dim, "rank": rank, "alpha": alpha, "atrous": atrous,
            "dilations": dilations, "branch_channels": branch_channels,
        }
        self.q_adapter = LowRankPath(**kwargs)
        self.v_adapter = LowRankPath(**kwargs)
        # Respect models constructed directly on a particular device or dtype.
        self.q_adapter.to(device=base.weight.device, dtype=base.weight.dtype)
        self.v_adapter.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.base(x).split(self.dim, dim=-1)
        # Concatenation avoids in-place mutation of views from the fused output.
        return torch.cat((q + self.q_adapter(x), k, v + self.v_adapter(x)), dim=-1)
