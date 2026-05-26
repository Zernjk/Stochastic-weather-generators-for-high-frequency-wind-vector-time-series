import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WindModelLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.2):
        super(WindModelLSTM, self).__init__()

        self.lstm = nn.LSTM(
              input_size=input_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              batch_first=True,
              bidirectional=True  # Unidirectional is sufficient
          )

        self.fnn = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size*2)
        
        # Consider both the last output and the mean of all outputs
        # last_out = lstm_out[:, -1, :]  # Last timestep output
        mean_out = torch.mean(lstm_out, dim=1)  # Mean over all timesteps
        
        # Combine the two outputs (concatenate, add, or take one - here we'll take the mean output)
        combined_out = mean_out
        
        # Apply the feedforward neural network
        fnn_out = self.fnn(combined_out)
        
        # Apply sigmoid activation for binary classification
        output = self.sigmoid(fnn_out)
        
        return output
    