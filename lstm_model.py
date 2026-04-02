import torch
import torch.nn as nn


class LSTMCalibrationModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.0
    ):
        super(LSTMCalibrationModel, self).__init__()

        # ==============================
        # LSTM LAYER
        # ==============================
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # ==============================
        # FULLY CONNECTED LAYERS
        # ==============================
        """self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )"""
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """

        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        out = self.fc(last_output)

        return out
        