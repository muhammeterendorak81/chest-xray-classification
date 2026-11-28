# Göğüs Röntgeni ile Hastalık Sınıflandırması

Bu proje Uygulamalı Yapay Zeka dersi için yaptığım dönem projesi. Göğüs röntgeni görüntülerine bakıp kişinin sağlıklı mı, pnömoni mi ya da COVID-19 mu olduğunu tahmin eden modeller eğitiyorum.

## Veri setleri
- Chest X-Ray Images (Pneumonia), Kermany ve ark.: https://data.mendeley.com/datasets/rscbjbr9sj/3 (Kaggle kopyası: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- COVID-19 Radiography Database, Chowdhury ve ark.: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

İki veri seti birleştirilmedi, ayrı ayrı çalıştım. Veri Seti 1'de iki aşama var: Normal/Pnömoni ve Bakteriyel/Viral. Veri Seti 2'de 4 sınıf var.

## Kurulum
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

Veri setlerini indirip `data/chest_xray` ve `data/covid19_radiography` klasörlerine açmak gerekiyor (veri GitHub'da yok).

## Çalıştırma sırası
1. `notebooks/01_eda.ipynb`: veri analizi
2. `notebooks/02_preprocessing.ipynb`: ön işleme ve veri bölmeleri (`splits/` klasörüne kaydedilir)
3. `notebooks/03_modelleme.ipynb`: Veri Seti 2 modelleri
4. `notebooks/04_veri1_modelleme.ipynb`: Veri Seti 1 modelleri
5. `notebooks/05_karsilastirma.ipynb`: sonuç tabloları ve grafikler

Kodlar `src/` klasöründe, sonuçlar `sonuclar/` klasöründe.

## Sonuçlar (test macro F1)
| | MLP | Basit CNN | DenseNet121 |
|---|---|---|---|
| Veri 2 (4 sınıf) | 0,771 | 0,881 | 0,951 |
| Veri 1, Aşama 1 (Normal/Pnömoni) | 0,792 | 0,765 | 0,898 |
| Veri 1, Aşama 2 (Bakteriyel/Viral) | 0,840 | 0,899 | 0,858 |

