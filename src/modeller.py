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
def basit_cnn(sinif_sayisi):
    def blok(giris, cikis):
        return nn.Sequential(
            nn.Conv2d(giris, cikis, kernel_size=3, padding=1),
            nn.BatchNorm2d(cikis),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    return nn.Sequential(
        blok(3, 16),
        blok(16, 32),
        blok(32, 64),
        blok(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, sinif_sayisi),
    )
from torchvision import models


def densenet121(sinif_sayisi):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, sinif_sayisi)
    return m