import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tahmin_et(model, yukleyici_):
    model.eval()
    gercek, tahmin = [], []
    with torch.no_grad():
        for x, y in yukleyici_:
            cikti = model(x.to(CIHAZ))
            tahmin.extend(cikti.argmax(1).cpu().tolist())
            gercek.extend(y.tolist())
    return np.array(gercek), np.array(tahmin)


def egit(model, egitim_y, dogrulama_y, sinif_agirliklari, epoch=20, lr=1e-3, sabir=5):
    model = model.to(CIHAZ)
    kayip = nn.CrossEntropyLoss(weight=sinif_agirliklari.to(CIHAZ))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    en_iyi_f1, en_iyi_agirlik, bekle = 0.0, None, 0
    gecmis = []
    for e in range(1, epoch + 1):
        model.train()
        toplam = 0.0
        for x, y in egitim_y:
            x, y = x.to(CIHAZ), y.to(CIHAZ)
            opt.zero_grad()
            l = kayip(model(x), y)
            l.backward()
            opt.step()
            toplam += l.item() * len(y)
        egitim_kaybi = toplam / len(egitim_y.dataset)
        g, t = tahmin_et(model, dogrulama_y)
        f1 = f1_score(g, t, average="macro")
        acc = accuracy_score(g, t)
        gecmis.append({"epoch": e, "egitim_kaybi": egitim_kaybi, "dog_acc": acc, "dog_f1": f1})
        print(f"epoch {e}: kayıp {egitim_kaybi:.4f} | doğrulama doğruluk {acc:.3f} | F1 {f1:.3f}")
        if f1 > en_iyi_f1:
            en_iyi_f1, en_iyi_agirlik, bekle = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bekle += 1
            if bekle >= sabir:
                print("erken durduruldu")
                break
    model.load_state_dict(en_iyi_agirlik)
    return model, gecmis