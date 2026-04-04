"""
model.py
RNN-SGRU with Self-Attention for Fraud Detection
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):           # x: (batch, seq, hidden)
        weights = torch.softmax(self.attn(x), dim=1)   # (batch, seq, 1)
        return (weights * x).sum(dim=1)                # (batch, hidden)


class RNN_SGRU(nn.Module):
    """
    Simplified GRU + Attention fraud classifier.

    Input  : (batch, n_features)  — one tabular row per sample
    Output : (batch, 1)           — raw logit

    Use sigmoid on output for probabilities.
    Use BCEWithLogitsLoss during training (more stable than BCELoss).
    """

    def __init__(self, input_size=12, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (batch, 1, features)
        gru_out, _ = self.gru(x)        # (batch, 1, hidden)
        context    = self.attention(gru_out)   # (batch, hidden)
        context    = self.dropout(context)
        return self.classifier(context) # (batch, 1) — raw logit
