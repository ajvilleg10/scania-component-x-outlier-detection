from __future__ import annotations

import math

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for multivariate time-series reconstruction."""

    def __init__(self, n_features: int, hidden_dim: int = 128, latent_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]
        z = self.to_latent(h)
        decoded_seed = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(decoded_seed)
        return self.output_layer(decoded)


class CNNLSTMAutoencoder(nn.Module):
    """CNN-LSTM Autoencoder for local and temporal pattern reconstruction."""

    def __init__(self, n_features: int, conv_channels: int = 32, hidden_dim: int = 128, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(conv_channels, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: batch, time, features
        x_conv = self.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        x_conv = self.dropout(x_conv)
        _, (h_n, _) = self.encoder(x_conv)
        z = self.to_latent(h_n[-1])
        seed = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(seed)
        return self.output_layer(decoded)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time windows."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerLiteAutoencoder(nn.Module):
    """Simplified Transformer Encoder Autoencoder for controlled Colab experiments.

    This is not a full TranAD or Anomaly Transformer implementation. It is a
    lightweight, regularized Transformer Encoder used for a resource-controlled
    comparison in the TFM.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 500,
        use_bottleneck: bool = True,
        bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(d_model, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, d_model),
            )
        else:
            self.bottleneck = nn.Identity()
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x):
        z = self.input_projection(x)
        z = self.positional_encoding(z)
        encoded = self.encoder(z)
        encoded = self.bottleneck(encoded)
        return self.output_projection(encoded)
