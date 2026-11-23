import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
from egitim import CIHAZ


def olc(model, test_y, siniflar):
    model.eval()
    gercek, olasilik = [], []
    with torch.no_grad():
        for x, y in test_y:
            p = torch.softmax(model(x.to(CIHAZ)), dim=1).cpu().numpy()
            olasilik.append(p)
            gercek.extend(y.tolist())
    olasilik = np.vstack(olasilik)
    gercek = np.array(gercek)
    tahmin = olasilik.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(gercek, tahmin, average="macro", zero_division=0)
    if olasilik.shape[1] == 2:
        auc = roc_auc_score(gercek, olasilik[:, 1])
    else:
        auc = roc_auc_score(gercek, olasilik, multi_class="ovr")
    sonuc = {
        "accuracy": float(accuracy_score(gercek, tahmin)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "roc_auc": float(auc),
    }
    kar = pd.DataFrame(confusion_matrix(gercek, tahmin), index=siniflar, columns=siniflar)
    return sonuc, kar


def kaydet(ad, model, gecmis, sonuc, kar):
    klasor = Path("../sonuclar")
    klasor.mkdir(exist_ok=True)
    pd.DataFrame(gecmis).to_csv(klasor / f"{ad}_gecmis.csv", index=False)
    kar.to_csv(klasor / f"{ad}_karisiklik.csv")
    (klasor / f"{ad}_metrik.json").write_text(json.dumps(sonuc, indent=2))
    torch.save(model.state_dict(), klasor / f"{ad}.pth")