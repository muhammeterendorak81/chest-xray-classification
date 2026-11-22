import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ORT = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

egitim_donusum = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(ORT, STD),
])

degerlendirme_donusum = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(ORT, STD),
])


class RontgenVeri(Dataset):
    def __init__(self, csv_yolu, bolum, etiket_sutunu="sinif", donusum=None, filtre=None):
        tum = pd.read_csv(csv_yolu)
        if filtre is not None:
            tum = tum[tum[filtre[0]] == filtre[1]]
        self.siniflar = sorted(tum[etiket_sutunu].unique())
        self.df = tum[tum["bolum"] == bolum].reset_index(drop=True)
        self.etiket_sutunu = etiket_sutunu
        self.donusum = donusum

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        satir = self.df.iloc[i]
        img = Image.open(satir["yol"])
        if self.donusum is not None:
            img = self.donusum(img)
        etiket = self.siniflar.index(satir[self.etiket_sutunu])
        return img, etiket


def yukleyici(csv_yolu, bolum, etiket_sutunu="sinif", filtre=None, batch=32, egit=False):
    donusum = egitim_donusum if egit else degerlendirme_donusum
    veri = RontgenVeri(csv_yolu, bolum, etiket_sutunu, donusum, filtre)
    return DataLoader(veri, batch_size=batch, shuffle=egit, num_workers=0)