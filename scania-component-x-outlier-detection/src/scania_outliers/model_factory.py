from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from scania_outliers.models.autoencoders import (
    CNNLSTMAutoencoder,
    LSTMAutoencoder,
    TransformerLiteAutoencoder,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str


class ModelFactory:
    """Factory for the three model families used in the TFM.

    Public command names use the same terminology as the dissertation:
    - lstm_autoencoder
    - cnn_lstm_autoencoder
    - transformer_encoder

    The old name `transformer_encoder_simplified` is still accepted as an alias
    so older commands do not break, but all outputs are stored with the canonical
    `transformer_encoder` name.
    """

    SUPPORTED = {
        "lstm_autoencoder": ModelSpec("lstm_autoencoder", "Recurrent reconstruction baseline."),
        "cnn_lstm_autoencoder": ModelSpec("cnn_lstm_autoencoder", "Hybrid local-temporal reconstruction model."),
        "transformer_encoder": ModelSpec("transformer_encoder", "Simplified Transformer encoder autoencoder."),
    }
    ALIASES = {
        "transformer_encoder_simplified": "transformer_encoder",
    }

    @classmethod
    def canonical_name(cls, model_name: str) -> str:
        return cls.ALIASES.get(model_name, model_name)

    @classmethod
    def create(cls, model_name: str, n_features: int, window_size: int, config: dict[str, Any]) -> nn.Module:
        model_name = cls.canonical_name(model_name)
        if model_name not in cls.SUPPORTED:
            raise ValueError(f"Unsupported model '{model_name}'. Valid: {list(cls.SUPPORTED)}")

        modeling = config.get("modeling", {})
        hidden_dim = int(modeling.get("hidden_dim", 128))
        latent_dim = int(modeling.get("latent_dim", 64))

        if model_name == "lstm_autoencoder":
            return LSTMAutoencoder(
                n_features=n_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=float(modeling.get("dropout", 0.0)),
            )

        if model_name == "cnn_lstm_autoencoder":
            return CNNLSTMAutoencoder(
                n_features=n_features,
                conv_channels=int(modeling.get("conv_channels", 32)),
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=float(modeling.get("dropout", 0.1)),
            )

        tr = modeling.get("transformer_encoder", modeling.get("transformer_encoder_simplified", {}))
        return TransformerLiteAutoencoder(
            n_features=n_features,
            d_model=int(tr.get("d_model", 64)),
            nhead=int(tr.get("nhead", 4)),
            num_layers=int(tr.get("num_layers", 2)),
            dim_feedforward=int(tr.get("dim_feedforward", 128)),
            dropout=float(tr.get("dropout", 0.1)),
            max_len=max(int(window_size), int(tr.get("max_len", 500))),
            use_bottleneck=bool(tr.get("use_bottleneck", True)),
            bottleneck_dim=int(tr.get("bottleneck_dim", 32)),
        )

    @classmethod
    def resolve_requested_models(cls, requested: str, config: dict[str, Any]) -> list[str]:
        requested = cls.canonical_name(requested)
        if requested == "all":
            configured = config.get("modeling", {}).get("models", list(cls.SUPPORTED))
            resolved = []
            for model_name in configured:
                canonical = cls.canonical_name(model_name)
                if canonical in cls.SUPPORTED and canonical not in resolved:
                    resolved.append(canonical)
            return resolved
        if requested not in cls.SUPPORTED:
            raise ValueError(f"Unsupported model '{requested}'. Valid: all or {list(cls.SUPPORTED)}")
        return [requested]
