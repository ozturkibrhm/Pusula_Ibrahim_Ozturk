# README

**İbrahim Öztürk**  
📧 ibrahim.ozturk1502@gmail.com  

---

## 📌 Proje Özeti

Bu proje, **Fiziksel Tıp & Rehabilitasyon veri seti** üzerinde kapsamlı **Exploratory Data Analysis (EDA)** ve **Ön İşleme (Pre-Processing)** adımlarını kapsamaktadır.  
Amaç, `TedaviSuresi` hedef değişkeni etrafında veri setini **temiz, tutarlı ve modellemeye hazır** hale getirmektir.  

Proje kapsamında:  
- Veri kaynağının incelenmesi  
- Eksik değerlerin analizi ve doldurulması  
- Çoklu değerli kolonların standardizasyonu  
- Anomalilerin ve duplikelerin tespiti  
- Kategorik alanların encoding ile sayısallaştırılması  
- Model öncesi en uygun **TOP-K özellik seçimleri** (20, 25, 30) karşılaştırması  
- **En iyi TOP-K = 25** sonucuyla final dataset üretilmesi  

**Nihai çıktı:** `dataset_model_ready_best_top25.(xlsx|csv)` ✅  

---

## 📂 Dosya Yapısı

```
├── Clean_Data_Case_DT_2025_std.xlsx           # Standardize edilmiş veri
├── Clean_Data_Case_DT_2025_clean.xlsx         # Temizlenmiş veri
├── Clean_Data_Case_DT_2025_clean_final.*      # Dedup/anomali temizlenmiş
├── dataset_gender_filled.*                    # Cinsiyet doldurma sonrası
├── dataset_imputed.*                          # İmputasyon (eksik doldurma)
├── dataset_final_imputed.*                    # Final imputasyon sonrası
├── dataset_model_ready_best_top25.*           # Model-ready (TOP-K=25)
└── notebooks/                                 # EDA ve preprocessing notebookları
```

---

## ▶️ Çalıştırma Talimatları

### 1) Gereksinimler
- Python 3.9+  
- Jupyter Notebook veya Kaggle ortamı  
- Gerekli kütüphaneler:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl rapidfuzz
```

### 2) Veri Kaynağı
- Orijinal dosya:  
  `/kaggle/input/talent-academy-case-dt-2025/Talent_Academy_Case_DT_2025.xlsx`  

### 3) Adım Adım Çalıştırma
1. Notebook veya scripti açın.  
2. Veri setini `input` klasöründen okuyun.  
3. Aşağıdaki adımları sırasıyla çalıştırın:  
   - **EDA:** Dağılımlar, korelasyonlar, hedef ilişkileri  
   - **Ön İşleme:** Sürelerin sayısallaştırılması, eksik değerlerin işlenmesi  
   - **Temizlik:** Anomali & duplikelerin çıkarılması  
   - **Doldurma:** Cinsiyet, kan grubu, alerji, kronik hastalık alanları  
   - **Encoding:** OHE ve Top-K token seçimi  
4. Çalışma sonunda **final dataset** üretilecektir.  

### 4) Çıktılar
- `Clean_Data_Case_DT_2025_std.xlsx`  
- `Clean_Data_Case_DT_2025_clean.xlsx`  
- `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`  
- `dataset_gender_filled.(xlsx|csv)`  
- `dataset_final_imputed.(xlsx|csv)`  
- **`dataset_model_ready_best_top25.(xlsx|csv)`** ✅  

---

## 📊 Örnek Kullanım

Python ortamında veri yükleme ve hızlı inceleme:  

```python
import pandas as pd

# Final dataset yükleme
df = pd.read_excel("dataset_model_ready_best_top25.xlsx")

# İlk 5 satır
print(df.head())

# Özet istatistikler
print(df.describe())
```

Grafik örneği:  

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["tedavi_suresi"], kde=True)
plt.title("Tedavi Süresi Dağılımı")
plt.show()
```

---

## 📈 Proje Akışı

1. **Veri Yükleme** → Excel’den DataFrame’e aktarma  
2. **EDA** → Dağılımlar, korelasyonlar, hedef ilişkileri  
3. **Ön İşleme** → Standardizasyon, NaN analizi  
4. **Temizlik** → Anomali & duplikelerin çıkarılması  
5. **Doldurma (Imputation)** → Hasta bazlı ve grup bazlı kurallar  
6. **Encoding & Top-K Seçimi** → Kategorik alanların sayısallaştırılması  
7. **Model-Ready Dataset** → En iyi TOP-K seçimiyle tek final dataset  

---

