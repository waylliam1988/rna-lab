"""Fixed-strength prior-logit offsets, isolated from production training CLI."""
from __future__ import annotations

import math
import torch


def add_prior_logits(residual, probabilities, *, alpha, epsilon=1e-4):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError('epsilon must lie strictly between zero and one half')
    # Exact ablation: no clipping, arithmetic, prior lookup, or RNG consumption.
    if alpha == 0:
        return residual
    if probabilities is None or probabilities.shape != residual.shape:
        raise ValueError('Prior and residual must have identical batch-length shapes')
    if not torch.isfinite(probabilities).all() or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Prior must contain finite probabilities in [0,1]')
    prior = probabilities.detach().to(device=residual.device, dtype=residual.dtype)
    return residual + alpha * torch.logit(prior.clamp(epsilon, 1-epsilon))


class ThermoResidualModel(torch.nn.Module):
    """Same CNN parameters; alpha/epsilon fixed by experiment configuration."""
    def __init__(self, cnn, store, *, alpha, epsilon=1e-4):
        super().__init__()
        # Validate constants even when no forward pass is performed.
        add_prior_logits(torch.zeros(1), None if alpha == 0 else torch.zeros(1),
                         alpha=alpha, epsilon=epsilon)
        self.cnn = cnn
        self.store = store
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def forward(self, tokens):
        residual = self.cnn(tokens)
        prior = self.store.batch(tokens) if self.alpha else None
        return add_prior_logits(residual, prior, alpha=self.alpha, epsilon=self.epsilon)
