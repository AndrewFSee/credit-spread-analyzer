"""
Deep-learning models for the Credit Spread Analysis & Prediction Platform.

Provides a PyTorch Dataset with sliding windows, LSTM and Transformer encoder
models, and a unified training loop with early stopping.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CreditSpreadDataset:
    """Sliding-window dataset for credit-spread time-series.

    Wraps ``torch.utils.data.Dataset`` to create (sequence, label) pairs using
    a fixed look-back window.

    Parameters
    ----------
    X:
        Feature array of shape ``(n_samples, n_features)``.
    y:
        Target array of shape ``(n_samples,)``.
    seq_len:
        Number of time steps in each input sequence.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20) -> None:
        try:
            import torch
            from torch.utils.data import Dataset

            class _Inner(Dataset):
                def __init__(self_, X_: np.ndarray, y_: np.ndarray, seq_len_: int) -> None:
                    self_.X = torch.tensor(X_, dtype=torch.float32)
                    self_.y = torch.tensor(y_, dtype=torch.float32)
                    self_.seq_len = seq_len_

                def __len__(self_) -> int:
                    return len(self_.X) - self_.seq_len

                def __getitem__(self_, idx: int):  # type: ignore[override]
                    x_seq = self_.X[idx : idx + self_.seq_len]
                    target = self_.y[idx + self_.seq_len]
                    return x_seq, target

            self._dataset = _Inner(X, y, seq_len)
        except ImportError as exc:
            raise ImportError("PyTorch is required: pip install torch") from exc

    def __len__(self) -> int:
        """Return the number of available windows."""
        return len(self._dataset)

    def __getitem__(self, idx: int):
        """Return (sequence_tensor, target_tensor) for the given index."""
        return self._dataset[idx]

    def get_torch_dataset(self):  # type: ignore[return]
        """Return the inner torch Dataset object."""
        return self._dataset


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class LSTMModel:
    """Configurable LSTM regression/classification model (wraps nn.Module).

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    hidden_size:
        LSTM hidden state size.
    num_layers:
        Number of stacked LSTM layers.
    output_size:
        Number of output units (1 for regression/binary classification).
    dropout:
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ) -> None:
        try:
            import torch
            import torch.nn as nn

            class _LSTM(nn.Module):
                def __init__(self_) -> None:
                    super().__init__()
                    self_.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        batch_first=True,
                    )
                    self_.dropout = nn.Dropout(dropout)
                    self_.fc = nn.Linear(hidden_size, output_size)

                def forward(self_, x: "torch.Tensor") -> "torch.Tensor":
                    out, _ = self_.lstm(x)
                    out = self_.dropout(out[:, -1, :])  # take last time step
                    return self_.fc(out).squeeze(-1)

            self.model = _LSTM()
        except ImportError as exc:
            raise ImportError("PyTorch is required: pip install torch") from exc

    def parameters(self):  # type: ignore[return]
        """Expose model parameters for the optimizer."""
        return self.model.parameters()

    def __call__(self, x):  # type: ignore[return]
        """Forward pass."""
        return self.model(x)

    def eval(self):  # type: ignore[return]
        """Set model to evaluation mode."""
        return self.model.eval()

    def train(self, mode: bool = True):  # type: ignore[return]
        """Set model to training mode."""
        return self.model.train(mode)

    def state_dict(self):  # type: ignore[return]
        """Return model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[return]
        """Load model state dict."""
        return self.model.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class TransformerModel:
    """Transformer encoder model for credit-spread forecasting.

    Uses a sinusoidal positional encoding followed by stacked
    ``TransformerEncoderLayer`` blocks and a linear projection head.

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    d_model:
        Model embedding dimension (must be divisible by *nhead*).
    nhead:
        Number of attention heads.
    num_layers:
        Number of ``TransformerEncoderLayer`` blocks.
    output_size:
        Number of output units.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        try:
            import math

            import torch
            import torch.nn as nn

            class _Transformer(nn.Module):
                def __init__(self_) -> None:
                    super().__init__()
                    self_.input_proj = nn.Linear(input_size, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self_.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self_.fc = nn.Linear(d_model, output_size)
                    self_._d_model = d_model

                def _positional_encoding(self_, seq_len: int, device) -> "torch.Tensor":
                    pe = torch.zeros(seq_len, self_._d_model, device=device)
                    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, self_._d_model, 2, dtype=torch.float, device=device)
                        * (-math.log(10000.0) / self_._d_model)
                    )
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    return pe.unsqueeze(0)  # (1, seq_len, d_model)

                def forward(self_, x: "torch.Tensor") -> "torch.Tensor":
                    x = self_.input_proj(x)
                    x = x + self_._positional_encoding(x.size(1), x.device)
                    x = self_.transformer(x)
                    return self_.fc(x[:, -1, :]).squeeze(-1)

            self.model = _Transformer()
        except ImportError as exc:
            raise ImportError("PyTorch is required: pip install torch") from exc

    def parameters(self):  # type: ignore[return]
        """Expose model parameters."""
        return self.model.parameters()

    def __call__(self, x):  # type: ignore[return]
        """Forward pass."""
        return self.model(x)

    def eval(self):  # type: ignore[return]
        """Set to eval mode."""
        return self.model.eval()

    def train(self, mode: bool = True):  # type: ignore[return]
        """Set to training mode."""
        return self.model.train(mode)

    def state_dict(self):  # type: ignore[return]
        """Return state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[return]
        """Load state dict."""
        return self.model.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dl_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lstm",
    seq_len: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 10,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    """Train an LSTM or Transformer model with early stopping.

    Parameters
    ----------
    X:
        Feature array of shape ``(n_samples, n_features)``.
    y:
        Target array of shape ``(n_samples,)``.
    model_type:
        ``"lstm"`` or ``"transformer"``.
    seq_len:
        Sliding window length.
    hidden_size:
        Hidden/embedding dimension.
    num_layers:
        Number of stacked layers.
    epochs:
        Maximum training epochs.
    lr:
        Learning rate.
    batch_size:
        Mini-batch size.
    patience:
        Early-stopping patience (epochs without val-loss improvement).
    val_fraction:
        Fraction of data held out for validation (taken from the end, no shuffling).

    Returns
    -------
    dict
        Keys: ``model``, ``train_losses``, ``val_losses``, ``predictions``,
        ``metrics``.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required: pip install torch") from exc

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    n = len(X_arr)
    input_size = X_arr.shape[1]

    # Train / val split (temporal – no shuffle)
    val_start = int(n * (1 - val_fraction))
    X_train, X_val = X_arr[:val_start], X_arr[val_start:]
    y_train, y_val = y_arr[:val_start], y_arr[val_start:]

    # Build datasets
    dataset_train = CreditSpreadDataset(X_train, y_train, seq_len).get_torch_dataset()
    dataset_val = CreditSpreadDataset(X_val, y_val, seq_len).get_torch_dataset()

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Instantiate model
    if model_type == "lstm":
        wrapped = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "transformer":
        d_model = hidden_size
        nhead = min(4, d_model)
        while d_model % nhead != 0:
            nhead -= 1
        wrapped = TransformerModel(
            input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'lstm' or 'transformer'.")

    model_nn = wrapped.model
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict = {}
    train_losses: list[float] = []
    val_losses: list[float] = []
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model_nn.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model_nn(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / max(len(dataset_train), 1))

        # ---- Validation ----
        model_nn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model_nn(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
        val_losses.append(val_loss / max(len(dataset_val), 1))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            logger.info(
                "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch,
                epochs,
                train_losses[-1],
                val_losses[-1],
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    # Restore best weights
    if best_state:
        model_nn.load_state_dict(best_state)

    # Generate predictions on full validation set
    predictions, metrics = evaluate_dl_model(wrapped, X_val, y_val, seq_len=seq_len)

    return {
        "model": wrapped,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "predictions": predictions,
        "metrics": metrics,
    }


def evaluate_dl_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 20,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate predictions and compute metrics for a trained DL model.

    Parameters
    ----------
    model:
        Trained ``LSTMModel`` or ``TransformerModel`` instance.
    X:
        Feature array of shape ``(n_samples, n_features)``.
    y:
        Target array of shape ``(n_samples,)``.
    seq_len:
        Sliding window length used during training.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        ``(predictions, metrics)`` where *predictions* is a 1-D array aligned
        to observations from index *seq_len* onward.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required: pip install torch") from exc

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    dataset = CreditSpreadDataset(X_arr, y_arr, seq_len).get_torch_dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model_nn = model.model
    model_nn.eval()
    preds_list: list[np.ndarray] = []

    with torch.no_grad():
        for xb, _ in loader:
            out = model_nn(xb)
            preds_list.append(out.cpu().numpy())

    predictions = np.concatenate(preds_list)
    y_aligned = y_arr[seq_len : seq_len + len(predictions)]

    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore

    rmse = float(np.sqrt(mean_squared_error(y_aligned, predictions)))
    mae = float(mean_absolute_error(y_aligned, predictions))
    dir_acc = float(np.mean(np.sign(y_aligned) == np.sign(predictions)))

    metrics: dict[str, float] = {"rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}
    return predictions, metrics
