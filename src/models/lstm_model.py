"""
GlucoTwin: Multimodal Glucose Spike Prediction - PyTorch LSTM with Temporal Attention
=====================================================================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Deep learning architecture for fusing continuous dynamic sensor telemetry
(interstitial glucose kinetics, circadian cycles, heart rate variability, physical activity)
with static Electronic Health Records (EHR demographics, clinical biomarkers, medications)
to forecast 2-hour forward glucose spikes (>180 mg/dL) for Type 2 Diabetes patients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("glucotwin.models.lstm")


class TemporalAttention(nn.Module):
    """
    Bahdanau-style self-attention mechanism over recurrent hidden states.

    Learns an attention weight distribution across lookback time-steps,
    amplifying critical events (e.g., rapid postprandial glucose momentum,
    sudden drop in physical activity) while suppressing steady-state noise.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_outputs : torch.Tensor
            Tensor of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        context_vector : torch.Tensor
            Attended context representation of shape (batch_size, hidden_dim).
        attention_weights : torch.Tensor
            Normalized softmax weights of shape (batch_size, seq_len).
        """
        # Energy scores: (batch_size, seq_len, 1) -> squeeze to (batch_size, seq_len)
        scores = self.projection(lstm_outputs).squeeze(-1)
        # Softmax along sequence dimension
        attention_weights = torch.softmax(scores, dim=-1)
        # Compute weighted sum context: (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_dim)
        # -> (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context_vector, attention_weights


class GlucoseLSTM(nn.Module):
    """
    Multimodal Dual-Branch Neural Architecture for Glucose Spike Prediction.

    Branch 1 (Dynamic/Temporal):
        Input shape: (batch_size, 24 timesteps, 54 features)
        - 2-layer stacked LSTM (hidden_size=128, dropout=0.3)
        - Self-attention pooling over 24 lookback steps -> 128-dim context

    Branch 2 (Static/Clinical):
        Input shape: (batch_size, 46 static EHR features)
        - Dense MLP: Linear(46, 64) -> ReLU -> Linear(64, 64) -> ReLU -> 64-dim representation

    Multimodal Fusion:
        - Concatenation: 128 (attended temporal) + 64 (static EHR) = 192 dimensions

    Classification Head:
        - Linear(192, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
        - Outputs continuous probability P(glucose > 180 mg/dL within 2 hours).
    """

    def __init__(
        self,
        temporal_dim: int = 54,
        static_dim: int = 46,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        static_mlp_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.static_mlp_dim = static_mlp_dim
        self.dropout_rate = dropout

        # 1. Recurrent Temporal Branch
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # 2. Temporal Self-Attention
        self.attention = TemporalAttention(
            hidden_dim=hidden_dim,
            attention_dim=64,
        )

        # 3. Static EHR MLP Branch: Linear(46) -> ReLU -> Linear(64)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, static_mlp_dim),
            nn.ReLU(),
            nn.Linear(static_mlp_dim, static_mlp_dim),
            nn.ReLU(),
        )

        # 4. Multimodal Fusion Dimension: 128 + 64 = 192
        fusion_dim = hidden_dim + static_mlp_dim

        # 5. Classification Head:
        # Linear(192) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Linear(1) -> Sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward computation pass.

        Parameters
        ----------
        x_temporal : torch.Tensor
            Batch of temporal lookback windows of shape (batch, 24, 54).
        x_static : torch.Tensor
            Batch of static patient characteristics of shape (batch, 46).
        return_attention : bool, default=False
            If True, returns tuple of (spike_probability, attention_weights).

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Predicted spike probability of shape (batch, 1) in [0, 1].
            Optionally attention weights of shape (batch, 24).
        """
        # 1. Recurrent sequence encoding
        lstm_out, _ = self.lstm(x_temporal)  # (batch, seq_len, 128)

        # 2. Self-attention over recurrent steps
        attended_temporal, attention_weights = self.attention(lstm_out)  # (batch, 128), (batch, 24)

        # 3. Static feature representation
        static_rep = self.static_mlp(x_static)  # (batch, 64)

        # 4. Multimodal concatenation: (batch, 192)
        fused = torch.cat([attended_temporal, static_rep], dim=-1)

        # 5. Spike risk output: (batch, 1)
        prob = self.classifier(fused)

        if return_attention:
            return prob, attention_weights
        return prob

    def get_attention_weights(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Extracts temporal attention weights across the lookback horizon.

        Parameters
        ----------
        x_temporal : np.ndarray or torch.Tensor
            Temporal sequence tensor of shape (batch, 24, 54) or (24, 54).
        x_static : np.ndarray or torch.Tensor, optional
            Static patient characteristics.
        device : torch.device, optional
            Target device for computation.

        Returns
        -------
        np.ndarray
            Attention weights array of shape (batch, 24) or (24,).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        squeeze_needed = False
        if isinstance(x_temporal, np.ndarray):
            if x_temporal.ndim == 2:
                x_temporal = np.expand_dims(x_temporal, axis=0)
                squeeze_needed = True
            t_temp = torch.as_tensor(x_temporal, dtype=torch.float32, device=device)
        else:
            t_temp = x_temporal.to(device)
            if t_temp.ndim == 2:
                t_temp = t_temp.unsqueeze(0)
                squeeze_needed = True

        with torch.no_grad():
            lstm_out, _ = self.lstm(t_temp)
            _, attention_weights = self.attention(lstm_out)
            weights_np = attention_weights.detach().cpu().numpy()

        if squeeze_needed:
            return weights_np[0]
        return weights_np

    def predict_proba(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Compute predicted spike probability for batches of arbitrary size.

        Parameters
        ----------
        x_temporal : np.ndarray or torch.Tensor
            Input temporal sequences, shape (N, 24, 54).
        x_static : np.ndarray or torch.Tensor
            Input static features, shape (N, 46).
        device : torch.device, optional
            Compute device.
        batch_size : int, default=1024
            Batch size for evaluation.

        Returns
        -------
        np.ndarray
            Predicted spike probabilities, shape (N,).
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        is_single = False
        if isinstance(x_temporal, np.ndarray) and x_temporal.ndim == 2:
            x_temporal = np.expand_dims(x_temporal, axis=0)
            is_single = True
        if isinstance(x_static, np.ndarray) and x_static.ndim == 1:
            x_static = np.expand_dims(x_static, axis=0)

        n_samples = len(x_temporal)
        probs_list = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                batch_t = (
                    torch.as_tensor(x_temporal[i:end_i], dtype=torch.float32, device=device)
                    if isinstance(x_temporal, np.ndarray)
                    else x_temporal[i:end_i].to(device)
                )
                batch_s = (
                    torch.as_tensor(x_static[i:end_i], dtype=torch.float32, device=device)
                    if isinstance(x_static, np.ndarray)
                    else x_static[i:end_i].to(device)
                )

                batch_probs = self.forward(batch_t, batch_s)
                probs_list.append(batch_probs.detach().cpu().numpy())

        all_probs = np.vstack(probs_list).flatten()
        if is_single:
            return float(all_probs[0])
        return all_probs

    def predict(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Binary prediction of glucose spike within 2 hours.

        Parameters
        ----------
        x_temporal : np.ndarray or torch.Tensor
            Shape (N, 24, 54).
        x_static : np.ndarray or torch.Tensor
            Shape (N, 46).
        threshold : float, default=0.5
            Decision boundary probability.

        Returns
        -------
        np.ndarray
            Binary predictions array (0 or 1), shape (N,).
        """
        probs = self.predict_proba(x_temporal, x_static, device=device)
        if np.isscalar(probs):
            return int(probs >= threshold)
        return (probs >= threshold).astype(int)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model architecture parameters and weights to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "temporal_dim": self.temporal_dim,
                "static_dim": self.static_dim,
                "hidden_dim": self.hidden_dim,
                "num_lstm_layers": self.num_lstm_layers,
                "static_mlp_dim": self.static_mlp_dim,
                "dropout": self.dropout_rate,
            },
        }
        torch.save(checkpoint, target_path)
        logger.info("Saved GlucoseLSTM checkpoint to %s", target_path)

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "GlucoseLSTM":
        """Load trained GlucoseLSTM instance from checkpoint file."""
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {target_path}")

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        checkpoint = torch.load(target_path, map_location=device)
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info("Loaded GlucoseLSTM checkpoint from %s onto %s", target_path, device)
        return model
