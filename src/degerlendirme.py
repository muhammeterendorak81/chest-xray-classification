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
import matplotlib.pyplot as plt
import seaborn as sns


def grafik_ciz(ad):
    klasor = Path("../sonuclar")
    gecmis = pd.read_csv(klasor / f"{ad}_gecmis.csv")
    kar = pd.read_csv(klasor / f"{ad}_karisiklik.csv", index_col=0)

    fig, eksen = plt.subplots(1, 2, figsize=(10, 4))
    eksen[0].plot(gecmis["epoch"], gecmis["egitim_kaybi"])
    eksen[0].set_title(f"{ad}: eğitim kaybı")
    eksen[0].set_xlabel("epoch")
    eksen[1].plot(gecmis["epoch"], gecmis["dog_f1"], label="doğrulama F1")
    eksen[1].plot(gecmis["epoch"], gecmis["dog_acc"], label="doğrulama doğruluk")
    eksen[1].set_title(f"{ad}: doğrulama başarısı")
    eksen[1].set_xlabel("epoch")
    eksen[1].legend()
    plt.tight_layout()
    plt.savefig(klasor / f"{ad}_egri.png", dpi=150)
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(kar, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{ad}: karışıklık matrisi (test)")
    plt.ylabel("gerçek")
    plt.xlabel("tahmin")
    plt.tight_layout()
    plt.savefig(klasor / f"{ad}_karisiklik.png", dpi=150)
    plt.show()