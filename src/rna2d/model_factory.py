"""Build prediction models from explicit, backward-compatible checkpoint configs."""
from __future__ import annotations

from torch import nn

from .model import PairBaseline
from .sequence_encoders import SequencePairModel


def normalize_model_config(config: dict) -> dict:
    architecture = config.get("architecture")
    kind = config.get("kind")
    if architecture in {"pair_baseline", "rna2d.model.PairBaseline"} or kind == "pair_baseline":
        hidden = config.get("hidden")
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("PairBaseline model config requires a positive hidden size")
        return {"architecture": "pair_baseline", "hidden": hidden}
    if kind in {"cnn", "rwkv_lite", "transformer", "egnn_seed"}:
        width, hidden, depth = config.get("width"), config.get("hidden"), config.get("depth")
        if not all(isinstance(value, int) and value > 0 for value in (width, hidden, depth)):
            raise ValueError("SequencePairModel config requires positive width, hidden and depth")
        return {
            "architecture": "sequence_pair_model",
            "kind": kind,
            "width": width,
            "hidden": hidden,
            "depth": depth,
        }
    raise ValueError(f"Unsupported or missing model architecture: {architecture!r}/{kind!r}")


def model_config_from_checkpoint(checkpoint: dict) -> dict:
    """Resolve modern configs and known historical checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    candidates = [
        checkpoint.get("model_config"),
        checkpoint.get("config", {}).get("model_config")
        if isinstance(checkpoint.get("config"), dict) else None,
        checkpoint.get("config", {}).get("model")
        if isinstance(checkpoint.get("config"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return normalize_model_config(candidate)
    if all(key in checkpoint for key in ("kind", "width", "hidden", "depth")):
        return normalize_model_config(checkpoint)
    legacy_hidden = checkpoint.get("hidden")
    if legacy_hidden is None and isinstance(checkpoint.get("config"), dict):
        legacy_hidden = checkpoint["config"].get("hidden")
    if isinstance(legacy_hidden, int) and legacy_hidden > 0:
        return {"architecture": "pair_baseline", "hidden": legacy_hidden}
    raise ValueError("Checkpoint does not contain a supported model configuration")


def build_pair_model(config: dict) -> nn.Module:
    normalized = normalize_model_config(config)
    if normalized["architecture"] == "pair_baseline":
        return PairBaseline(normalized["hidden"])
    return SequencePairModel(
        normalized["kind"], normalized["width"], normalized["hidden"], normalized["depth"]
    )


def model_from_checkpoint(checkpoint: dict) -> nn.Module:
    return build_pair_model(model_config_from_checkpoint(checkpoint))
