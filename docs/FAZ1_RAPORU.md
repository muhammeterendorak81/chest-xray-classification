# YAPAY ZEKA DÖNEM PROJESİ
## FAZ 1 – PROJE İLERLEME DURUM RAPORU

**Proje Başlığı:** Göğüs Röntgeni Görüntülerinden Derin Öğrenme ile Solunum Yolu Hastalığı Sınıflandırması  
**Öğrenci Adı Soyadı:** Muhammet Eren DORAK  
**Öğrenci Numarası:** 241230021  
**Ders Sorumlusu:** Dr. Öğr. Üyesi Hayri İncekara  
**Teslim Tarihi:** 09.11.2026  
**Proje Alanı:** Derin Öğrenme / Görüntü İşleme (Tıbbi Görüntü Sınıflandırma)  
**GitHub Deposu:** [https://github.com/muhammeterendorak81/chest-xray-classification](https://github.com/muhammeterendorak81/chest-xray-classification)

---

## 1. PROJENİN TANIMI VE AMACI

### 1.1. Problem Tanımı
Pnömoni (zatürre), akciğerlerdeki hava keseciklerinin bakteri, virüs ya da başka mikroorganizmalar nedeniyle iltihaplanmasıyla ortaya çıkan bir solunum yolu hastalığıdır. Dünya Sağlık Örgütü'ne göre pnömoni, dünya genelinde çocuklarda enfeksiyona bağlı ölümlerin en büyük tek nedenidir (WHO, 2022). COVID-19 salgını da viral pnömoninin ne kadar hızlı yayılabildiğini ve sağlık sistemine nasıl bir yük bindirdiğini göstermiştir.

Göğüs röntgeni bu hastalıkların tanısında en sık kullanılan görüntüleme yöntemidir: ucuzdur, hızlıdır ve hemen her sağlık kuruluşunda bulunur. Ancak röntgenin yorumlanması deneyimli bir radyolog gerektirir. Bakteriyel ve viral pnömoni ya da COVID-19 pnömonisi ile diğer akciğer opasiteleri röntgende birbirine benzer görünebilir. Uzman sayısının yetersiz olduğu yerlerde ve yoğun dönemlerde bu durum tanının gecikmesine veya hatalı konmasına yol açabilir.

Bu projede ele alınan problem, bir göğüs röntgeni görüntüsünden kişinin sağlıklı mı, yoksa pnömoni (bakteriyel/viral) ya da COVID-19 kaynaklı bir tablo mu olduğunu otomatik olarak tahmin eden bir derin öğrenme modeli geliştirmektir. Böyle bir model hekimin yerini almayı değil, ön tanı sürecinde ikinci bir görüş sunarak iş yükünü azaltmayı ve öncelikli hastaların daha hızlı fark edilmesini amaçlamaktadır.

### 1.2. Projenin Amacı
Projenin temel amacı, iki farklı açık veri setinde göğüs röntgeni sınıflandırması yapan modeller geliştirmek ve bu modellerin başarısını hem kendi aralarında hem de literatürle karşılaştırmaktır. Hedefler şunlardır:
1. **Veri Seti 1 (Chest X-Ray Images):** Kermany vd. (2018) ile uyumlu iki aşamalı sınıflandırma. 1. aşamada Normal / Pnömoni, 2. aşamada pnömoni görüntüleri arasında Bakteriyel / Viral ayrımı.
2. **Veri Seti 2 (COVID-19 Radiography):** Normal, COVID-19, Viral Pnömoni ve Lung Opacity olmak üzere dört sınıflı sınıflandırma.
3. **Model Karşılaştırması:** Her veri setinde, biri görüntü verisi için bilerek seçilmiş uygun olmayan bir referans model olmak üzere 3 farklı model eğitmek (MLP, Basit CNN, DenseNet121).
4. **Değerlendirme:** Sonuçları doğruluk, duyarlılık (recall), makro F1 skoru ve AUC ile ölçmek ve literatürdeki çalışmalarla kıyaslamak.

### 1.3. Projenin Kapsamı
**Kapsam dahilinde olanlar:**
- Yalnızca ön-arka (frontal) göğüs röntgeni görüntüleri kullanılacaktır.
- İki veri seti birleştirilmeyecek, her biri kendi sınıf yapısıyla ayrı ayrı ele alınacaktır.
- Keşifsel veri analizi, ön işleme, 3 model × 2 veri seti eğitimi (PyTorch ile), değerlendirme ve literatürle karşılaştırma yapılacaktır.

**Kapsam dışında kalanlar / sınırlar:**
- BT, MR gibi diğer görüntüleme türleri ve hastaya ait klinik bilgiler (yaş, şikâyet, laboratuvar) kullanılmayacaktır.
- Veri Seti 2'deki akciğer maskeleri incelenmiş, ancak segmentasyon problemi kapsam dışında bırakılmıştır.
- Geliştirilecek model bir klinik tanı aracı değildir; gerçek hastane verisiyle dış doğrulama yapılmayacaktır.

---

## 2. VERİ SETLERİNİN BULUNMASI VE ANALİZİ

### 2.1. Kullanılan Veri Setleri

#### Veri Seti 1 – Chest X-Ray Images (Pneumonia)
- **Adı:** Chest X-Ray Images (Pneumonia)
- **Kaynağı:** Kermany vd. (2018), Cell. Guangzhou Kadın ve Çocuk Sağlığı Merkezi'nde 1–5 yaş arası pediatrik hastalardan alınmıştır.
- **Erişim:** [Kaggle Bağlantısı](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Toplam Örnek:** 5.856 JPEG görüntü – 1.583 Normal, 4.273 Pnömoni (2.780 bakteriyel, 1.493 viral).
- **Orijinal Bölme:** Eğitim 5.216 / Doğrulama 16 / Test 624.

#### Veri Seti 2 – COVID-19 Radiography Database
- **Adı:** COVID-19 Radiography Database
- **Kaynağı:** Chowdhury vd. (2020) ve Rahman vd. (2021). PadChest, SIRM, GitHub, Kaggle ve RSNA kaynaklarından derlenmiştir.
- **Erişim:** [Kaggle Bağlantısı](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Toplam Örnek:** 21.165 PNG görüntü – 10.192 Normal, 6.012 Lung Opacity, 3.616 COVID-19, 1.345 Viral Pnömoni.
- **Bölme:** Hazır bölme yoktur; tarafımızca stratified olarak %70 Eğitim / %15 Doğrulama / %15 Test ayrılmıştır.

---

### 2.2. Veri Setinin Genel Özellikleri Tablosu

| İncelenen Özellik | Veri Seti 1: Chest X-Ray (Kermany vd.) | Veri Seti 2: COVID-19 Radiography |
| :--- | :--- | :--- |
| **Toplam veri sayısı** | 5.856 görüntü | 21.165 görüntü |
| **Eğitim verisi sayısı** | 5.216 (orijinal) → Yeniden bölme sonrası **4.447** | Hazır bölme yok → %70 Eğitim (**14.815**) |
| **Doğrulama verisi sayısı** | 16 (orijinal) → Yeniden bölme sonrası **785** | %15 Doğrulama (**3.175**) |
| **Test verisi sayısı** | **624** (orijinal test seti aynen korundu) | %15 Test (**3.175**) |
| **Sınıf sayısı** | 2 ana sınıf (Normal / Pnömoni) + Pnömonide 2 alt sınıf (Bakteriyel / Viral) | 4 sınıf (Normal, COVID-19, Lung Opacity, Viral Pnömoni) |
| **Sınıflara göre dağılım** | Normal: 1.583 (%27,0), Bakteriyel: 2.780 (%47,5), Viral: 1.493 (%25,5) | Normal: 10.192 (%48,2), Lung Opacity: 6.012 (%28,4), COVID: 3.616 (%17,1), Viral: 1.345 (%6,4) |
| **Eksik veri durumu** | Eksik etiket yok, bozuk dosya yok *(5.856 `__MACOSX` metadata dosyası temizlendi)* | Eksik etiket yok, bozuk dosya yok |
| **Etiket durumu** | Klasör adından Normal/Pnömoni; dosya adından Bakteriyel/Viral | Sınıf klasör adından (COVID, Normal, Lung_Opacity, Viral Pneumonia) |
| **Veri formatı** | JPEG | PNG |
| **Boyutlar / Özellikler** | Değişken: Genişlik 384–2.916 px, Yükseklik 127–2.713 px (Ortalama ~1328×971). 5.573 Gri (L), 283 RGB | **Sabit 299×299 piksel (std: 0)**. 21.025 Gri (L), 140 RGB |

---

### 2.3. Veri Setinin Görselleştirilmesi
Görseller projenin `notebooks/01_eda.ipynb` dosyası çalıştırılarak üretilmiş ve `gorseller/` dizinine kaydedilmiştir:
- **Görsel 1:** Sınıf Dağılım Grafiği (`gorseller/sinif_dagilimlari.png`)
- **Görsel 2:** Chest X-Ray Orijinal Bölüm Dağılımı ve 16'lık Val Kümesi Sorunu (`gorseller/bolum_dagilimi.png`)
- **Görsel 3:** Görüntü Boyutları Saçılım Grafiği (`gorseller/boyut_dagilimi.png`)
- **Görsel 4:** Piksel Parlaklık Histogramı (`gorseller/piksel_histogramlari.png`)

---

### 2.4. Veri Setinin Değerlendirilmesi
- **Yeterlilik:** Veri Seti 1 transfer öğrenme için yeterlidir (%92-96 doğruluk literatür standardıdır). Veri Seti 2 (21.165 görüntü) dört sınıf için zengin bir hacme sahiptir.
- **Sınıf Dengesizliği:** Belirgin dengesizlik vardır (Chest X-Ray 2,7:1; COVID-19'da Normal/Viral oranı 7,6:1). Kayıp fonksiyonunda dengelenmiş sınıf ağırlıkları (Class Weights) ve Makro F1 metriği kullanılacaktır.
- **Bozuk Dosya ve Kalite Kontrolü:** 27.021 görüntü doğrulanmış; Kaggle arşivinden gelen 5.856 sahte macOS metadata dosyası elenerek veri bütünlüğü sağlanmıştır.
- **Kısa Yol Öğrenme (Shortcut) Riski:** Normal röntgenlerin ortalamada daha büyük olması ve bazı görüntülerin RGB olması modele kısa yol sunabilir; bu nedenle tüm görüntüler 224×224 gri tona ve ardından 3 kanala eşitlenmiştir.

---

## 3. VERİ ÖN İŞLEME VE HAZIRLIK

### 3.1. Uygulanan Ön İşleme İşlemleri
1. **Görüntü Boyutlandırma:** Tüm görüntüler `transforms.Resize((224, 224))` ile sabit boyuta getirilmiştir.
2. **Kanal Standartlaştırması:** Görüntüler 3 kanallı RGB tensör formatına eşitlenmiştir.
3. **Piksel Normalizasyonu:** [0, 1] ölçeğine getirilerek ImageNet ortalama ve standart sapması (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) ile normalize edilmiştir.
4. **Tıbbi Veri Artırma:** Kalp anatomik olarak solda yer aldığından **yatay çevirme (horizontal flip) kullanılmamıştır.** Yalnızca ±10° döndürme uygulanmıştır.
5. **Veri Bölme (Data Splitting):**
   - **Chest X-Ray:** 16 görüntülük yetersiz val seti yerine, train seti stratified olarak %85 Eğitim (4.447) ve %15 Doğrulama (785) olarak yeniden yapılandırılmış; 624 görüntülük resmi Test seti aynen korunmuştur.
   - **COVID-19:** Stratified olarak %70 Eğitim (14.815), %15 Doğrulama (3.175), %15 Test (3.175) olarak bölünmüştür.
6. **Sınıf Ağırlıkları (Class Weights):**
   - COVID-19: Viral Pneumonia (3.932), COVID (1.463), Lung Opacity (0.880), Normal (0.519)
   - Chest X-Ray: NORMAL (1.939), PNEUMONIA (0.674)

---

### 3.2. Ön İşleme Öncesi ve Sonrası Durum Tablosu

| Özellik | Ön İşleme Öncesi | Ön İşleme Sonrası |
| :--- | :--- | :--- |
| **Veri Dağılımı (Chest X-Ray)** | 5.856 (5.216 / 16 / 624) | **5.856 (4.447 / 785 / 624)** |
| **Veri Dağılımı (COVID-19)** | 21.165 (Bölünmemiş) | **21.165 (14.815 / 3.175 / 3.175)** |
| **Veri Boyutu** | Değişken ve 299×299 | **Her iki veri seti: 224×224×3** |
| **Normalizasyon** | 0–255 tam sayı | [0, 1] float32 tensör (ImageNet normalize) |
| **Split Kayıtları** | Ham klasörler | `data/covid_split.csv` ve `data/xray_split.csv` |

---

## 4. LİTERATÜR TARAMASI

### 4.1. İlgili Akademik Çalışmalar Tablosu

| No | Makale / Çalışma Adı | Yıl | Veri Seti | Yöntem | Temel Sonuç | DOI |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning *(Kermany vd.)* | 2018 | Chest X-Ray (5.856 röntgen) | Transfer Öğrenme (Inception V3) | Normal/Pnömoni %92,8; Bakteriyel/Viral %90,7 | [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010) |
| **2** | CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays *(Rajpurkar vd. / Stanford)* | 2017 | NIH ChestX-ray14 (112.120 röntgen) | DenseNet121 | F1 = 0,435 ile radyolog ortalamasını (0,387) aştı | [10.48550/arXiv.1711.05225](https://doi.org/10.48550/arXiv.1711.05225) |
| **3** | An Efficient Deep Learning Approach to Pneumonia Classification *(Stephen vd.)* | 2019 | Chest X-Ray (Kermany) | Sıfırdan küçük CNN + Veri artırma | %93,73 doğrulama doğruluğu | [10.1155/2019/4180949](https://doi.org/10.1155/2019/4180949) |
| **4** | A Novel Transfer Learning Approach for Pneumonia Detection *(Chouhan vd.)* | 2020 | Chest X-Ray (Kermany) | 5 modelin topluluğu (Ensemble) | %96,4 doğruluk, %99,62 duyarlılık | [10.3390/app10020559](https://doi.org/10.3390/app10020559) |
| **5** | Can AI Help in Screening Viral and COVID-19 Pneumonia? *(Chowdhury vd.)* | 2020 | COVID-19 Radiography | 8 ön eğitimli CNN (en iyisi DenseNet) | 2 sınıfta %99,7, 3 sınıfta %97,9 | [10.1109/ACCESS.2020.3010287](https://doi.org/10.1109/ACCESS.2020.3010287) |

---

## 5. KULLANILACAK YÖNTEM VE MİMARİLER

### 5.1. Yöntem Seçimi
Modeller PyTorch (CUDA GPU hızlandırmalı) kütüphanesi ile kodlanacaktır.

| Model | Türü | Kullanılma Amacı | Seçilme Nedeni |
| :--- | :--- | :--- | :--- |
| **Model 1: MLP** | Uygun Olmayan Referans Model | Alt sınır (baseline) belirlemek | Pikselleri düzleştirdiği için uzamsal komşuluğu kaybeder. CNN'in gerekliliğini sayısal olarak ispatlamak için bilerek seçilmiştir. |
| **Model 2: Basit CNN** | Sıfırdan Eğitim | Hafif ve veri setine özel model | Stephen vd. (2019) yaklaşımı. Az parametreli, hızlı ve yorumlanabilir. |
| **Model 3: DenseNet121** | Transfer Öğrenme + İnce Ayar | En yüksek başarı beklenen ana model | Stanford CheXNet mimarisi. Tıbbi lezyonları yakalamada tıp literatürünün altın standardıdır. |

### 5.2. Değerlendirme Metrikleri
- **Accuracy (Doğruluk):** Literatür kıyası için.
- **Recall (Duyarlılık):** Hastayı kaçırmama başarısı (en kritik tıbbi metrik).
- **Makro F1-Skoru:** Sınıf dengesizliğinde azınlık sınıfların adil değerlendirilmesi için.
- **ROC-AUC ve Karışıklık Matrisi:** Karar eşiğinden bağımsız ayırt etme gücü ve sınıf karışıklıklarının analizi.

---

## 6. FAZ 1 UYGULAMA DURUMU VE ÇIKTILAR

| Gerçekleştirilen İş | Durum | Açıklama |
| :--- | :---: | :--- |
| **Veri setinin bulunması ve indirilmesi** | Tamamlandı | Kermany (2018) ve COVID-19 Radiography veri setleri indirildi (`data/`). |
| **Veri setinin analiz edilmesi** | Tamamlandı | `notebooks/01_eda.ipynb` ile boyut, renk modu, bozuk dosya taraması tamamlandı. |
| **Veri görselleştirmeleri** | Tamamlandı | `gorseller/` klasöründe 4 temel analitik grafik üretildi. |
| **Ön işleme işlemleri** | Tamamlandı | `notebooks/02_preprocessing.ipynb` ile stratified bölme, normalizasyon ve sınıf ağırlıkları hazırlandı. |
| **Literatür taraması** | Tamamlandı | 5 temel çalışma incelendi ve tablolandı. |
| **Yöntem / mimari seçimi** | Tamamlandı | MLP, Basit CNN ve DenseNet121 mimarileri belirlendi. |

**GitHub Depo Bağlantısı:**  
👉 [https://github.com/muhammeterendorak81/chest-xray-classification](https://github.com/muhammeterendorak81/chest-xray-classification)

---

## 7. SONUÇ VE SONRAKİ FAZIN PLANI

### 7.1. Karşılaşılan Problemler ve Çözümler
1. **16 Görüntülük Doğrulama Seti:** Train ile birleştirilip %15 ayrılarak 785 görüntülük güvenilir bir validasyon seti oluşturuldu.
2. **Bozuk Dosya Taramasında Yakalanan `__MACOSX` Dosyaları:** 5.856 sahte sistem dosyası temizlendi, gerçek verilerin sağlam olduğu kanıtlandı.
3. **Sınıf Dengesizliği:** Dengelenmiş sınıf ağırlıkları hesaplandı.
4. **Boyut Farklılıkları:** Tüm görüntüler 224×224 standardına getirildi.
5. **Tıbbi Yön Kuralı:** Kalbin sol tarafta olması nedeniyle yatay çevirme yapılmadı.

### 7.2. Faz 2 Planı
- 3 modelin PyTorch ile kodlanması ve DataLoader yapılarının kurulması.
- Veri Seti 1'de 1. Aşama (Normal/Pnömoni) ve 2. Aşama (Bakteriyel/Viral); Veri Seti 2'de 4 sınıflı model eğitimleri.
- Sınıf ağırlıkları, Erken Durdurma (EarlyStopping) ve DenseNet121 ince ayarı (Fine-Tuning).
- Test kümelerinde Accuracy, Recall, Makro F1, ROC-AUC ve Karışıklık Matrisi ile değerlendirme.
