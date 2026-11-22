import torch.nn as nn


def mlp(sinif_sayisi):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((64, 64)),
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, sinif_sayisi),
    )