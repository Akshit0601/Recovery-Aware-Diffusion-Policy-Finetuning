"""
MLP Value Critic Network for DPPO.

Approximates the state value function V_ϕ(s_t) where s_t ∈ R³⁸ is the stacked
observation conditioning vector (T_o = 2, D_obs = 19).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence, Optional


class ValueCritic(nn.Module):
    """
    Multi-layer Perceptron (MLP) Critic predicting scalar state-value V_ϕ(s).
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "mish",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        act_cls = nn.Mish if activation.lower() == "mish" else nn.ReLU

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_cls())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization for stable value estimation."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Final layer with gain 1.0 or smaller for value scale
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute V_ϕ(s).

        Args:
            state: Tensor of shape (..., input_dim)

        Returns:
            Tensor of shape (..., 1) or flattened if appropriate
        """
        orig_shape = state.shape[:-1]
        flat_state = state.view(-1, self.input_dim)
        val = self.net(flat_state)
        if len(orig_shape) == 0:
            return val.squeeze()
        return val.view(*orig_shape)
