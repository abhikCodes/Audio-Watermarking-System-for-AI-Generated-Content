import torch
import torch.nn as nn

class MothEncoder(nn.Module):
    """
    Learns a low-level residual Δx that, when scaled by α and added to
    the input audio, embeds a watermark but stays imperceptible.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, padding=7), nn.ReLU(),
            nn.Conv1d(16, 32, 15, groups=4, padding=7), nn.ReLU(),
            nn.Conv1d(32, 32, 15, groups=8, padding=7), nn.ReLU(),
            nn.Conv1d(32, 1, 15, padding=7), nn.Tanh()   # bound Δx
        )

    def forward(self, x):
        delta = self.net(x) * self.alpha
        return torch.clamp(x + delta, -1.0, 1.0)

class BatDetector(nn.Module):
    """
    Classifies whether an audio clip contains the watermark.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, padding=7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 15, padding=7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 15, padding=7), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x)) 