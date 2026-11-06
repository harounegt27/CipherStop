"""CipherStop - LSTM + Attention model for early ransomware detection.

Input : a batch of behavioral sequences, shape (batch, seq_len, n_features),
        plus the true length of each sequence (the rest is padding).
Output: a detection score in [0, 1] per sequence (1 = ransomware).
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    """Additive attention over the LSTM outputs.

    For each time step t:  e_t = v^T tanh(W h_t)
    Weights: alpha = softmax(e) over the *real* (non-padded) steps only.
    Context vector: sum_t alpha_t * h_t
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H), mask: (B, T) True for real steps
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T), padding -> 0
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, weights


class RansomwareLSTM(nn.Module):
    """LSTM + attention binary classifier for variable-length sequences.

    The LSTM is unidirectional on purpose: for early detection the model
    must only use the past, exactly like in real-time monitoring.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        attn_dim: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def logits(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw scores before sigmoid (use these with BCEWithLogitsLoss)."""
        B, T, _ = x.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        lengths = lengths.to("cpu").clamp(min=1, max=T)

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.to(
            x.device
        ).unsqueeze(1)
        context, weights = self.attention(out, mask)
        logit = self.classifier(self.dropout(context)).squeeze(-1)  # (B,)
        return logit, weights

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns detection score in [0, 1], shape (B,)."""
        logit, weights = self.logits(x, lengths)
        score = torch.sigmoid(logit)
        return (score, weights) if return_attention else score
