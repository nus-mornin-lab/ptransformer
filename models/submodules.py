import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, d_in, d_out) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, d_out)
        )

    def forward(self, x):

        logits = self.fc(x)

        return logits

