"""
DeepSpeech2 architecture implementation
"""

import torch
from torch import nn


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x, _ = self.rnn(x)
        B, T, H = x.shape
        x = self.batch_norm(x.reshape(B * T, H)).reshape(B, T, H)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DeepSpeech2(nn.Module):
    def __init__(
        self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, rnn_dropout=0.1
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )
        self._conv_layers = [m for m in self.conv if isinstance(m, nn.Conv2d)]

        conv_out_freq = self._transform_length(n_feats, axis=0)
        rnn_input_size = 32 * conv_out_freq

        self.rnn_layers = nn.ModuleList(
            [
                BatchRNN(
                    input_size=rnn_input_size if i == 0 else rnn_hidden * 2,
                    hidden_size=rnn_hidden,
                    dropout=rnn_dropout if i < rnn_layers - 1 else 0.0,
                )
                for i in range(rnn_layers)
            ]
        )

        self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

    def _transform_length(self, length, axis):
        """
        Apply every conv layer's own kernel/stride/padding (axis 0 = frequency,
        axis 1 = time) to compute the resulting length along that axis.
        """
        for conv in self._conv_layers:
            kernel = conv.kernel_size[axis]
            stride = conv.stride[axis]
            padding = conv.padding[axis]
            length = (length + 2 * padding - kernel) // stride + 1
        return length

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)  # (B, 1, F, T)
        x = self.conv(x)  # (B, 32, F', T')

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)

        for layer in self.rnn_layers:
            x = layer(x)

        logits = self.fc(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        return self._transform_length(input_lengths, axis=1)

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
