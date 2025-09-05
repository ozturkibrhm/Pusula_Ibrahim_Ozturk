
# EDA
**Hazırlayan:** İbrahim Öztürk  
**Kapsam:** EDA bulguları (grafikler + yorumlar), ön işleme adımları (hangi sütunda ne yapıldı, neden), tekrar kullanılabilir yardımcı fonksiyonlar ve analiz kodları. Bu rapor, Kaggle Notebook ortamında yürütülen çalışmanın **yalnızca bu projede kullanılan kodlarını ve bulgularını** dokümante eder.

---

## 1) Veri Kaynağı ve Çalışma Ortamı
- **Kaynak dosya:** `/kaggle/input/talent-academy-case-dt-2025/Talent_Academy_Case_DT_2025.xlsx`
- **Kayıtlı çıktı (temiz veri):** `/kaggle/working/dataset_clean.xlsx`
- **Ortam:** Kaggle Python 3 (NumPy, Pandas, Matplotlib, Seaborn vb. yüklü)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import os

# Input data files are available in the read-only "../input/" directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# /kaggle/input/talent-academy-case-dt-2025/Talent_Academy_Case_DT_2025.xlsx
```

---

## 2) Kütüphaneler ve Görsel Ayarlar
- Uyarılar kapatıldı, sütun genişliği ve temel figür boyutları ayarlandı.

```python
# kütüphaneleri import edelim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import seaborn as sns

# pythonda uyarıları kapatır
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 200)
sns.set(rc={"figure.figsize": (8,5)})
```

---

## 3) Veri Yükleme ve İlk İnceleme
- Dosya okundu, sütun bilgileri ve özet istatistikler incelendi.

```python
# Kaggle yolunu kendine göre kontrol et
path = "/kaggle/input/talent-academy-case-dt-2025/Talent_Academy_Case_DT_2025.xlsx"
df = pd.read_excel(path)
df.head()
df.shape
df.info()
df.describe()

# veri içerisinde bulunan sütunları listeleyelim.
df.columns
```

**Gözlem:** 2.235 satır × 13 sütun. `Cinsiyet`, `KanGrubu`, `KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri` gibi kategorik/sözel alanlarda boşluklar mevcut.

---

## 4) Değişken İsimleri: Python Uyumlu Standart
**Neden?** Python tarafında okunabilirlik/tekrar kullanılabilirlik için `snake_case` tercih edildi.

```python
df = df.copy()
df.rename(columns={
    'HastaNo' : 'hasta_no',
    'Yas' : 'yas',
    'Cinsiyet' : 'cinsiyet',
    'KanGrubu' : 'kan_grubu',
    'Uyruk' : 'uyruk',
    'KronikHastalik' : 'kronik_hastalik',
    'Bolum' : 'bolum',
    'Alerji' : 'alerji',
    'Tanilar' : 'tanilar',
    'TedaviAdi' : 'tedavi_adi',
    'TedaviSuresi' : 'tedavi_suresi',
    'UygulamaYerleri': 'uygulama_yerleri',
    'UygulamaSuresi' : 'uygulama_suresi'
}, inplace=True)
df.head()
```

---

## 5) Süre Alanlarını Sayısallaştırma (String → Int)
**Neden?** `tedavi_suresi` ve `uygulama_suresi` metin içinde **“Seans/Dakika”** gibi ifadelerle geldiği için sayısal analiz/modelleme öncesi tam sayı formatına dönüştürüldü.

```python
def to_nullable_int(series, round_mode="round"):
    x = (series.astype(str)
             .str.replace(",", ".", regex=False)
             .str.extract(r"(\d+(?:\.\d+)?)")[0]
             .astype(float))
    if round_mode == "round":
        x = x.round()
    elif round_mode == "floor":
        x = np.floor(x)
    elif round_mode == "ceil":
        x = np.ceil(x)
    return x.astype("int64")  # pandas (nullable değil) int64

# Uygulama:
if "tedavi_suresi" in df.columns:
    df["tedavi_suresi"] = to_nullable_int(df["tedavi_suresi"], round_mode="round")

if "uygulama_suresi" in df.columns:
    df["uygulama_suresi"] = to_nullable_int(df["uygulama_suresi"], round_mode="round")
```

**Çıktı Kaydetme (Temiz Veri):**
```python
out_path_xlsx = "/kaggle/working/dataset_clean.xlsx"
df.to_excel(out_path_xlsx, index=False)
df.head()
```

---

## 6) Eksik Değer Sayımı
- Hangi sütunda kaç adet eksik var?

```python
df.isnull().sum()
```

**Örnek bulgular:**  
- `cinsiyet`: 169, `kan_grubu`: 675, `kronik_hastalik`: 611, `alerji`: 944, `tanilar`: 75, `uygulama_yerleri`: 221 (diğerleri tam).

---

## 7) EDA Yardımcıları
- **Amaç:** NaN/boşları tek etiket altında toplamak; çoklu değerli alanları listelemeye uygun hâle getirmek; görselleri okunur kılmak.

```python
sns.set_theme()
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 120)

def fill_cat(s, null_label="Bilinmiyor"):
    """Grafikler/korelasyon için NaN ve boş stringleri tek etikete indir."""
    return (
        s.astype("string")
         .str.strip()
         .replace("", pd.NA)
         .fillna(null_label)
    )

def split_multi(x):
    """Virgüllü/Listedeki değerleri temiz listeye çevir."""
    if isinstance(x, list):
        return [t.strip() for t in x if str(t).strip() != ""]
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(",") if t.strip() != ""]

# >>> ÇİZİM YARDIMCILARI <<<
def plot_corr_bars(series, title):
    """Korelasyonları simetrik eksen ve etiketle gösteren barplot."""
    s = series.dropna().sort_values(key=lambda x: x.abs(), ascending=True)
    max_abs = float(np.ceil(s.abs().max() * 100) / 100) or 0.01
    fig_h = max(2.5, 0.4 * len(s) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(s.index, s.values)
    ax.axvline(0, ls="--", lw=1, color="k", alpha=.4)
    ax.set_xlim(-max_abs - 0.05, max_abs + 0.05)
    ax.set_title(title); ax.set_xlabel("Pearson corr"); ax.set_ylabel("")
    for y, v in enumerate(s.values):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:.2f}", va="center")
    plt.tight_layout(); plt.show()

def plot_corr_heatmap(series, title, top_n=15):
    """Tek satırlık heatmap'i kontrastlı ve okunur çizer (merkez=0)."""
    s = series.dropna().head(top_n)
    max_abs = float(np.ceil(s.abs().max() * 100) / 100) or 0.01
    data = s.to_frame("corr")
    w = max(6, 0.55 * len(s) + 2)
    plt.figure(figsize=(w, 2.8))
    sns.heatmap(
        data.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-max_abs, vmax=max_abs, cbar=False, linewidths=.5, linecolor="white"
    )
    plt.title(title); plt.yticks([]); plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.show()
```

---

## 8) Temel EDA (Dağılımlar, Kategorikler, Çoklu Değerler)
- **Sayısal dağılımlar:** `yas`, `tedavi_suresi`, `uygulama_suresi` için histogram/boxplot.
- **Kategorikler:** `cinsiyet`, `kan_grubu`, `uyruk`, `bolum`, `tedavi_adi`, `tanilar` için sayım grafikleri (top-15).
- **Çoklu değer kolonlar:** `kronik_hastalik`, `alerji`, `tanilar`, `uygulama_yerleri` explode edilerek dağılım + ortalama `tedavi_suresi` hesaplandı.

```python
df_plot = df.copy()

# 1) Sayısal dağılımlar
num_cols = [c for c in ["yas", "tedavi_suresi", "uygulama_suresi"] if c in df_plot.columns]
for c in num_cols:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_plot[c].dropna(), kde=True, ax=ax)
    ax.set_title(f"{c} dağılımı"); ax.set_xlabel(c)
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(4,4))
    sns.boxplot(y=df_plot[c], ax=ax)
    ax.set_title(f"{c} boxplot (aykırılar)"); ax.set_ylabel(c)
    plt.tight_layout(); plt.show()

# 2) Tekil kategorik dağılımlar (top-15)
cat_cols = [c for c in ["cinsiyet","kan_grubu","uyruk","bolum","tedavi_adi","tanilar"] if c in df_plot.columns]
for c in cat_cols:
    s = fill_cat(df_plot[c])
    vc = s.value_counts().head(15)
    display(pd.DataFrame({c: vc.index, "count": vc.values}))
    fig, ax = plt.subplots(figsize=(12,5))
    g = sns.countplot(x=s, order=vc.index, ax=ax)
    ax.set_title(f"{c} dağılımı (top-15)")
    ax.set_xlabel(c); ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    for container in g.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=9)
    ax.yaxis.set_ticks([])
    plt.tight_layout(); plt.show()

# 3) Çoklu-değer kolonlar: dağılım + (varsa) ortalama tedavi_suresi
multi_cols = [c for c in ["kronik_hastalik","alerji","tanilar","uygulama_yerleri"] if c in df_plot.columns]
for c in multi_cols:
    ex = df_plot.copy()
    ex[c] = ex[c].apply(split_multi)
    ex = ex.explode(c, ignore_index=False)
    ex[c] = fill_cat(ex[c])

    # COUNT (top-20)
    vc = ex[c].value_counts().head(20)
    display(pd.DataFrame({c: vc.index, "count": vc.values}))
    plt.figure(figsize=(10,6))
    sns.barplot(y=vc.index, x=vc.values, orient="h")
    plt.title(f"{c} dağılımı (top-20)")
    plt.xlabel("count"); plt.ylabel(c)
    plt.tight_layout(); plt.show()

    # Ortalama tedavi süresi (top-20 sıklığa göre)
    if "tedavi_suresi" in ex.columns:
        tab = (ex.groupby(c)["tedavi_suresi"]
                 .agg(count="count", mean="mean")
                 .sort_values("count", ascending=False)
                 .head(20))
        display(tab.round(2))
        plt.figure(figsize=(10,6))
        sns.barplot(y=tab.index, x=tab["mean"], orient="h")
        plt.title(f"{c} → Ortalama Tedavi Süresi (top-20)")
        plt.xlabel("Ortalama seans"); plt.ylabel(c)
        plt.tight_layout(); plt.show()
```

---

## 9) Hedefle İlişkiler ve Özetler
- **Yaş vs Tedavi Süresi** (scatter)
- **Cinsiyet / Kan Grubu ~ Ortalama Tedavi Süresi** (bar & boxplot)
- **Cinsiyet × Kan Grubu** ve **Cinsiyet × Yaş Grubu** pivot + ısı haritası
- **Sayısal korelasyon ısı haritası**

```python
# 4) Hedefle basit ilişkiler
if {"yas","tedavi_suresi"}.issubset(df_plot.columns):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x="yas", y="tedavi_suresi", data=df_plot, s=20, ax=ax)
    ax.set_title("Yaş vs Tedavi Süresi (seans)")
    ax.set_xlabel("Yaş"); ax.set_ylabel("Tedavi Süresi (seans)")
    plt.tight_layout(); plt.show()

for c in ["cinsiyet","kan_grubu"]:
    if {c,"tedavi_suresi"}.issubset(df_plot.columns):
        s = fill_cat(df_plot[c])
        grp = df_plot.groupby(s)["tedavi_suresi"].mean().sort_values()
        display(grp.round(2))
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=grp.index, y=grp.values, ax=ax)
        ax.set_title(f"{c} ~ Ortalama tedavi_suresi")
        ax.set_ylabel("Ortalama seans"); ax.set_xlabel(c)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.show()

# 4.1) Kan grubu → tedavi süresi
if {"kan_grubu","tedavi_suresi"}.issubset(df_plot.columns):
    kg = fill_cat(df_plot["kan_grubu"])
    tablo = (
        df_plot.assign(kan_grubu_=kg)
               .groupby("kan_grubu_")["tedavi_suresi"]
               .agg(count="count", mean="mean", median="median")
               .round(2)
               .sort_values("count", ascending=False)
    )
    display(tablo)

    plt.figure(figsize=(10,5))
    order = kg.value_counts().index
    sns.boxplot(x=kg, y=df_plot["tedavi_suresi"], order=order)
    plt.xticks(rotation=45)
    plt.title("Tedavi Süresi (seans) ~ Kan Grubu")
    plt.xlabel("Kan Grubu"); plt.ylabel("Tedavi Süresi (seans)")
    plt.tight_layout(); plt.show()

# 10) Cinsiyet × Kan grubu ve Cinsiyet × Yaş (bin) → ortalama tedavi süresi
if {"cinsiyet","kan_grubu","tedavi_suresi"}.issubset(df_plot.columns):
    ck = pd.pivot_table(
        df_plot.assign(cinsiyet_=fill_cat(df_plot["cinsiyet"]), kan_grubu_=fill_cat(df_plot["kan_grubu"])),
        index="cinsiyet_", columns="kan_grubu_", values="tedavi_suresi", aggfunc="mean"
    )
    display(ck.round(2))
    plt.figure(figsize=(8,5)); sns.heatmap(ck, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Cinsiyet × Kan Grubu → Ortalama Tedavi Süresi")
    plt.xlabel("Kan Grubu"); plt.ylabel("Cinsiyet")
    plt.tight_layout(); plt.show()

if {"cinsiyet","yas","tedavi_suresi"}.issubset(df_plot.columns):
    tmp = df_plot.copy()
    tmp["yas_grup"] = pd.cut(tmp["yas"], bins=[0,30,50,120], labels=["Genç","Orta","Yaşlı"], right=False)
    cy = pd.pivot_table(
        tmp.assign(cinsiyet_=fill_cat(tmp["cinsiyet"])),
        index="cinsiyet_", columns="yas_grup", values="tedavi_suresi", aggfunc="mean"
    )
    display(cy.round(2))
    plt.figure(figsize=(6,4)); sns.heatmap(cy, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Cinsiyet × Yaş Grubu → Ortalama Tedavi Süresi")
    plt.xlabel("Yaş Grubu"); plt.ylabel("Cinsiyet")
    plt.tight_layout(); plt.show()

# 5) Sayısal korelasyon
if len(num_cols) >= 2:
    corr = df_plot[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(4.8,4.2))
    sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1)
    plt.title("Sayısal Korelasyon")
    plt.tight_layout(); plt.show()
```

---

## 10) Korelasyon Paketleri (OHE / Token)
- **Tekil kategorikler (OHE) ~ hedef** korelasyon (Top-30 |abs|)
- **Çoklu-değer kolonlar** tokenlaştırılıp hedefle korelasyon (Top-N) + ısı haritası

```python
target = "tedavi_suresi"

# 12.1 Tekil kategorikler → OHE → hedefle korelasyon
single_cats = [c for c in ["cinsiyet","kan_grubu","uyruk","bolum","tedavi_adi"] if c in df_plot.columns]
single_cats = [c for c in single_cats if df_plot[c].notna().any()]

if single_cats and target in df_plot.columns:
    df_ohe_base = df_plot[[target]].copy()
    for c in single_cats:
        s = fill_cat(df_plot[c])
        d = pd.get_dummies(s, prefix=c, drop_first=False, dtype=int)
        df_ohe_base = df_ohe_base.join(d)

    corr_s = (df_ohe_base.drop(columns=[target])
                        .corrwith(df_ohe_base[target])
                        .sort_values(key=lambda x: x.abs(), ascending=False))
    top_k = min(30, len(corr_s))
    corr_top = corr_s.head(top_k)
    display(corr_top.to_frame("corr").round(3))

    plot_corr_bars(corr_top, "Tekil kategorikler ~ tedavi_suresi (korelasyon, Top-30)")

# 12.2 Çoklu-değer kolonlar → token 1/0 → hedefle korelasyon
multi_cols = [c for c in ["kronik_hastalik","tanilar","alerji","uygulama_yerleri"] if c in df_plot.columns]
for c in multi_cols:
    ex = df_plot[[target, c]].copy()
    ex[c] = ex[c].apply(split_multi)
    ex = ex.explode(c, ignore_index=False)
    ex[c] = fill_cat(ex[c])

    wide = (ex.assign(present=1)
              .pivot_table(index=ex.index, columns=c, values="present", aggfunc="max", fill_value=0))

    min_count = 20  # nadir token filtresi
    keep = [col for col, s in (wide.sum()).items() if s >= min_count and col != "Bilinmiyor"]
    if not keep:
        print(f"\n[{c}] Yeterli sıklıkta token bulunamadı (min_count={min_count}).")
        continue

    w2 = wide[keep].join(df_plot[target])
    corr_m = w2[keep].corrwith(w2[target]).sort_values(key=lambda x: x.abs(), ascending=False)
    top_k = min(25, len(corr_m))
    corr_top = corr_m.head(top_k)
    display(corr_top.to_frame("corr").round(3))

    plot_corr_bars(corr_top, f"{c} token ~ {target} (korelasyon, Top-{top_k})")
    plot_corr_heatmap(corr_m, f"{c} (Top-15) ~ {target} korelasyon heatmap", top_n=15)
```

---

## 11) Eksik Veri Analizi (Kan Grubu Örneği)
- Eksik/dolu gruplarda yaş ve cinsiyet dağılımları karşılaştırıldı.

```python
if "kan_grubu" in df_plot.columns:
    print("\n— Kan grubu EKSİK olanlarda Yaş dağılımı —")
    display(df_plot[df_plot["kan_grubu"].isna()]["yas"].describe())

    if "cinsiyet" in df_plot.columns:
        print("\n— Kan grubu EKSİK olanlarda Cinsiyet dağılımı —")
        display(df_plot[df_plot["kan_grubu"].isna()]["cinsiyet"].value_counts(dropna=False))

    print("\n— Kan grubu DOLU olanlarda Yaş dağılımı —")
    display(df_plot[df_plot["kan_grubu"].notna()]["yas"].describe())

    if "cinsiyet" in df_plot.columns:
        print("\n— Kan grubu DOLU olanlarda Cinsiyet dağılımı —")
        display(df_plot[df_plot["kan_grubu"].notna()]["cinsiyet"].value_counts(dropna=False))
```

---

## 12) Tedavi × Uygulama Yeri Isı Haritaları
- **COUNT** ve **Ortalama Uygulama Süresi** pivotları çizildi.

```python
if {"tedavi_adi","uygulama_yerleri"}.issubset(df_plot.columns):
    df_ex_u = df_plot.copy()
    df_ex_u["uygulama_yerleri"] = df_ex_u["uygulama_yerleri"].apply(split_multi)
    df_ex_u = df_ex_u.explode("uygulama_yerleri", ignore_index=False)
    df_ex_u["uygulama_yerleri"] = fill_cat(df_ex_u["uygulama_yerleri"])

    top_tedavi = fill_cat(df_plot["tedavi_adi"]).value_counts().head(20).index
    sub = df_ex_u[df_ex_u["tedavi_adi"].isin(top_tedavi)]

    # COUNT
    heat_count = (sub.groupby(["tedavi_adi","uygulama_yerleri"])
                    .size().unstack(fill_value=0))
    plt.figure(figsize=(12,8))
    sns.heatmap(heat_count, cmap="Blues")
    plt.title("Top-20 Tedavi Adı × Uygulama Yerleri (COUNT)")
    plt.xlabel("Uygulama Yeri"); plt.ylabel("Tedavi Adı")
    plt.tight_layout(); plt.show()

    # MEAN uygulama_suresi
    if "uygulama_suresi" in df_plot.columns:
        mean_pivot = (sub.groupby(["tedavi_adi","uygulama_yerleri"])["uygulama_suresi"]
                        .mean().unstack())
        plt.figure(figsize=(12,8))
        sns.heatmap(mean_pivot, cmap="Blues")
        plt.title("Tedavi Adı × Uygulama Yeri → Ortalama Uygulama Süresi")
        plt.xlabel("Uygulama Yeri"); plt.ylabel("Tedavi Adı")
        plt.tight_layout(); plt.show()
```

---

## 13) Tutarlılık ve Kalite Kontrolleri
- **Şüpheli Kayıt Örneği:** `Dorsalji -Boyun+trapez+skapular` tedavisi olup uygulama yerinde yalnızca `Boyun` yazılı satırların listelenmesi.
- **Olası Duplikeler:** Anahtar alan seti ile toplu tarama.
- **Basit Mantık Kontrolleri:** Yaş aralığı, `tedavi_suresi` / `uygulama_suresi` boşluk kontrolü.

```python
# Şüpheli satırlar
if {"tedavi_adi","uygulama_yerleri","uygulama_suresi"}.issubset(df_plot.columns):
    mask_susp = (
        df_plot["tedavi_adi"].astype("string").str.contains(r"boyun\+trapez\+skapular", case=False, na=False)
        & (df_plot["uygulama_yerleri"].astype("string").str.contains(",", na=False) == False)
        & (df_plot["uygulama_yerleri"].astype("string").str.strip().str.lower() == "boyun")
    )
    susp_rows = df_plot.loc[mask_susp, ["hasta_no","tedavi_adi","uygulama_yerleri","uygulama_suresi"]]
    display(susp_rows.head(50))

# Duplikeler
dup_keys = ["hasta_no","yas","cinsiyet","kan_grubu","uyruk",
            "kronik_hastalik","bolum","tanilar","tedavi_adi",
            "tedavi_suresi","uygulama_yerleri"]
dup_keys_exist = [k for k in dup_keys if k in df_plot.columns]
dupes = (df_plot[df_plot.duplicated(subset=dup_keys_exist, keep=False)]
         .sort_values(dup_keys_exist))
display(dupes.head(20))

# Basit mantık kontrolleri
checks = {}
if "yas" in df_plot.columns:
    checks["yas_out_of_range"] = int(((df_plot["yas"] < 0) | (df_plot["yas"] > 110)).sum())
if "tedavi_suresi" in df_plot.columns:
    checks["tedavi_suresi_null"] = int(df_plot["tedavi_suresi"].isna().sum())
if "uygulama_suresi" in df_plot.columns:
    checks["uygulama_suresi_null"] = int(df_plot["uygulama_suresi"].isna().sum())

pd.Series(checks)
```

---

## 14) Özet Bulgular (Kısa Notlar)
- **Eksikler:** `kan_grubu` (675), `alerji` (944) başta olmak üzere bazı alanlarda belirgin boşluklar var.
- **Kategorik çeşitlilik:** `kronik_hastalik` (~220 farklı ifade), `tanilar` (~367), `tedavi_adi` (~244), `uygulama_yerleri` (~37).
- **Metin standardizasyonu ihtiyacı:** Büyük/küçük harf, yazım hatası, varyantlar (örn. `Polen`, `POLEN`, `Toz`, `TOZ`, `Voltaren`, `VOLTAREN`) tek standarda indirilmeli (model öncesi encode için temiz temsil gerekir).
- **İlişki ipuçları:** Bölüm/tedavi adı gibi özelliklerin `tedavi_suresi` ile anlamlı korelasyonları gözlenebilir. Çoklu değer kolonlarda nadir token filtrelemesi (min_count=20) ile daha sağlam korelasyon görselleştirmeleri elde edildi.
- **Kalite:** Bazı tedavi–uygulama_yeri kombinasyonlarında (ör. sadece `Boyun`) tutarlılık kontrolü ihtiyacı tespit edildi. Olası duplikeler listelendi.
- **Sayısallaştırma:** `tedavi_suresi` ve `uygulama_suresi` doğru biçimde tam sayıya dönüştürüldü (birimler atılarak).

---

## 15) Sonuç ve Sonraki Adımlar
- **Bu rapor**, veri adlandırma standardizasyonu + sürelerin sayısallaştırılması, eksik değer durum analizi, kategorik/çoklu değer dağılımları ve hedefle başlangıç ilişkilerini içerir.
- **Model-Ready** aşaması için ek öneriler:
  1. Kategorik normalizasyon (case-folding, yazım düzeltme, eş-etiket birleştirme).
  2. Çoklu alanlarda token standardizasyonu ve nadir sınıfların birleştirilmesi.
  3. Gerekirse `uygulama_yerleri` çoklu satır patlatmayı kalıcılaştırıp **örnek düzeyinde** geniş tabloya pivotlama.
  4. Duplikelerin iş kurallarına göre tekilleştirilmesi (örn. aynı anahtar setinde `uygulama_suresi` ortalaması).
  5. Eksik değerlerin **anlamlı** doldurulması (ör. `kan_grubu` bilinmiyor etiketi gibi, modelde dummy ile).

> Not: Yukarıdaki kod ve bulgular doğrudan bu çalışmada kullanılan notebook’tan derlenmiştir.


# Veri Ön İşleme (Data Pre-Processing)

Bu rapor; **standartlaştırma → temizlik/deduplication → anomali ayıklama → grup konsolidasyonu → güvenli doldurma (imputation) → model-uyumlu encoding (Top‑K seçimi)** akışınızı **yalnızca paylaştığınız kod ve çıktılarına dayanarak** ayrıntılı biçimde dokümante eder.

---

## 1) Standartlaştırma (TR-duyarlı normalizasyon + ASCII opsiyonu)

**Amaç:** Çok-değerli metin alanlarındaki gürültüyü temizlemek, Türkçe karakterlere duyarlı normalize etmek, ASCII’ye katmanlı dönüşüm opsiyonu sağlamak ve fuzzy eşleştirme ile kanonik değerleri üretmek.

**İşlenen alanlar (çok-değerli):** `kronik_hastalik`, `alerji`, `tanilar`, `uygulama_yerleri`  
**Tek-değerli alanlar:** (şu sürümde boş, istenirse eklenebilir)

### 1.1. Temel fonksiyonlar
- `tr_lower`: Türkçe’ye duyarlı küçük harfe çevirme (`İ→i`, `I→ı`, ardından `.lower()`)
- `normalize_basic`: Boşluk sıkıştırma, noktalama sadeleştirme (TR harflerini korur)
- `to_ascii`: Diakritik temizliği + TR özgü harf eşlemeleri (örn. `ş→s`, `ç→c`)
- `fold_key`: Fuzzy karşılaştırma anahtarı (diakritik ve işaretsiz)
- `split_multi_cell` / `combine_tokens`: Çoklu hücre ayrıştırma ve set‑tabanlı yeniden birleştirme

### 1.2. Kanonik küme & fuzzy eşleme
- `build_canonical`: `min_count >= 5` olan normalize edilmiş aday terimler
- `make_mapper`: 3 aşamalı eşleme  
  1) **Manuel alias** (varsa)  
  2) **Doğrudan eşleşme** (kanonikte varsa)  
  3) **Fuzzy eşleme**: `rapidfuzz.fuzz.token_sort_ratio` ≥ **88**

> Çıktı iki dosya olarak üretilir:  
> - `Clean_Data_Case_DT_2025_std.xlsx` (orijinal + `*_std` kolonları)  
> - `Clean_Data_Case_DT_2025_clean.xlsx` (kolon **sırası korunur**, içerik standardize/ASCII)

**Doğrulama çıktıları:**  
- Kolon sırası korunumu: **True**  
- Standardize edilen kolonlar: `['kronik_hastalik','alerji','tanilar','uygulama_yerleri']`  
- ASCII çıktı: **Aktif**

---

## 2) Geniş Temizlik + Deduplication (Entegre Akış)

**Girdi:** `Clean_Data_Case_DT_2025_clean.xlsx`  
**Çıktı:** `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`

### 2.1. Kozmetik & anlamsal temizlik
- `tidy_trailing_punct`: Sondaki virgül/boşlukları temizleme
- `collapse_hyphen_repeat`: `x - x` tekrarlarını tekilleştirme
- `bolum` alanında **MSK (kas-iskelet) bağlamı** tespit edilirse “Solunum Merkezi” çıkarımı  
- `uygulama_yerleri`: ayrıştırma ve virgül/boşluk standardizasyonu

### 2.2. Taraf (sağ/sol) konsolidasyonu
- `detect_side`: `tedavi_adi` metninden taraf çıkarımı (regex)
- `enforce_side_on_site`: `uygulama_yerleri` içinde tarafı **uyumlu** hale getirme ve eksikse **ön ek** olarak ekleme (örn. `sol omuz bölgesi`)

### 2.3. Grup Anahtarı (GK) üretimi
- Türkçe karakter katlaması ve alfasayısal dışı karakterlerin temizlendiği `norm_key`
- Grup dışı bırakılan sütunlar: `{uygulama_yerleri, uygulama_suresi}`
- GK kolonları: özgün kolonların `_gk_*` sürümleri (gruplama için kullanılır, finalde atılır)

### 2.4. Hasta+Tedavi bazında uygulama_yerleri seçimi
- Sınıflandırma: **tüm vücut** vs **spesifik** bölge
- Kural: Mümkünse **spesifik** olan tercih edilir; spesifik içinde en **kapsamlı** (token sayısı yüksek) temsilci alınır

### 2.5. Global tutarlılık kuralları
- **Kural A:** Spesifik varsa **“Tüm Vücut”** kayıtları **aynı GK** içinde atılır  
- **Kural B:** **5 dk anomali** (eğer grupta `>5` dakikalık en az bir kayıt varsa, `5` olanlar atılır)

### 2.6. Tekilleştirme (GK bazında tek satıra indirgeme)
- Aday seçim:
  - `uygulama_yerleri`: **en kapsamlı** aday (en çok token)
  - `uygulama_suresi`: **mod** (eşitlikte **maksimum**)
  - Eğer en kapsamlı adaylarda mod yoksa: **mod’a en yakın süre** (eşitlikte küçük olan)
- Çıktıya yalnız **temsilci** satır yazılır; yardımcı kolonlar temizlenir

**Süreç özeti (log):**
```text
rows_in: 2235
dropped_whole_body: 4
dropped_duration_5: 351
rows_out: 511
✔ Kaydedildi: Clean_Data_Case_DT_2025_clean_final.xlsx / .csv
```

> Böylece 2000+ satırlık veri, **mantıklı gruplama ve kural setleri** ile **~500** satıra indirildi.

---

## 3) Cinsiyet Doldurma + Alerji/Kronik Normalizasyon

**Girdi:** `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`  
**Çıktı:** `dataset_gender_filled.(xlsx|csv)`

### 3.1. Alerji & Kronik NaN → "yok"
- Çoklu hücre ayrıştırma (`,`/`;`) ve **normalize** etme
- Örnek eşlemeler:  
  - `volteren → voltaren`  
  - `yer fistigi → yer fıstığı`  
  - `hiportiroidizm → hipotiroidizm`

### 3.2. Cinsiyet doldurma (kural + oransal)
- **Standartlaştırma:** `Kadın/Erkek` dışındaki varyantları normalize
- **Kural tabanlı çıkarım:** Tanı + kronik metinlerinde **kadın/erkek özgü anahtarlar**  
  - Kadın: `rahim, over, meme, gebelik, menopoz, ...`  
  - Erkek: `prostat, testis, varikosel, ...`
- **Oransal doldurma:** Kalan eksikler, veri içi oranlara göre (`rng=42` ile deterministik)

**Çıktı örneği:**
```text
[CINSIYET DAĞILIMI]
Kadın: 304
Erkek: 207
```

---

## 4) Güvenli Doldurma (Hasta → Grup → "bilinmiyor")

**Girdi:** `Clean_Data_Case_DT_2025_clean_final.xlsx`  
**Çıktı:** `dataset_imputed.(xlsx|csv)` ve entegre versiyonu `dataset_final_imputed.(xlsx|csv)`

### 4.1. Hedef kolonlar
- `kan_grubu`, `bolum`, `tanilar`, `uygulama_yerleri`

### 4.2. Doldurma stratejileri
1. **Hasta içi mutabakat:** Aynı `hasta_no` içinde tek değer varsa eksikleri onunla doldur
2. **Grup modu (güvenlik eşiğiyle):**
   - `bolum` ve `uygulama_yerleri`: Grup = (`tedavi_adi`,`tedavi_suresi`), `min_count≥3`, `mode_ratio≥0.7`
   - `tanilar`: Grup = (`tedavi_adi`,`tedavi_suresi`,`uygulama_yerleri`), **tam mutabakat** (`nunique=1`, `count≥5`)
3. **Fallback:** Hâlâ NaN ise **"bilinmiyor"**

**Özet çıktılar:**
```text
Başlangıç NaN: kan_grubu=136, bolum=2, tanilar=20/0, uygulama_yerleri=34
Hasta bazlı doldurulan: bolum=0, tanilar=4/0, uygulama_yerleri=11
Grup bazlı doldurulan:  bolum=0, tanilar=0, uygulama_yerleri=0
Final NaN: tüm hedef kolonlarda 0
```

> Örnek dağılımlar:
- **bolum (Top 10):** FTR 313 · FTR+Solunum 128 · Ortopedi 40 · …  
- **uygulama_yerleri (Top 10):** bel 104 · boyun 69 · tüm vücut bölgesi 47 · …

---

## 5) Model Uyumlu Encoding + Top‑K Seçimi (Tek Çıktı)

**Girdi:** `dataset_final_imputed.xlsx` (511 × 13)  
**Amaç:** Metin alanlarını **sayısal** hale getirip farklı **TOP_K** değerlerini çapraz doğrulama ile karşılaştırmak ve **en iyi K** için **tek** model-ready veri seti üretmek.

### 5.1. Özellik kümeleri
- **Sayısal:** `yas`, `tedavi_suresi` *(hedef)*, `uygulama_suresi`
- **Cinsiyet (binary):** `kadın=1`, `erkek=0`
- **One-Hot:** `kan_grubu`, `uyruk`, `bolum` *(drop_first=True)*
- **Multi-Label Tam OHE:** `uygulama_yerleri`, `alerji`, `kronik_hastalik`
- **Top‑K + other (multi-label):** `tanilar` → **TOP_K** en sık + `other`
- **Top‑K + other (tekil):** `tedavi_adi` → **TOP_K** en sık + `other` (tam OHE; drop_first=False)

> Tüm metinler `casefold()` ile normalize edilir, multi‑label alanlarda `[,;]` parçalama yapılır, NaN kalırsa 0 ile doldurulur.

### 5.2. Modeller & Değerlendirme
- **Model:** `RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)`  
- **CV:** `KFold(n_splits=5, shuffle=True, random_state=42)`  
- **Metrik:** `MAE` (negatif işareti çevrilerek raporlandı)

### 5.3. Sonuçlar (TOP_K ∈ {20, 25, 30})
| TOP_K | CV MAE (ort) | CV MAE (std) | Özellik sayısı |
|------:|:-------------:|:------------:|---------------:|
| 20 | 1.770 | 0.110 | 119 |
| **25** | **1.731** | **0.115** | **129** |
| 30 | 1.745 | 0.108 | 139 |

**Seçim:** En iyi TOP_K = **25**  
**Tek Çıktı:** `dataset_model_ready_best_top25.(xlsx|csv)`

---

## 6) Reprodüksiyon (Çalıştırma Sırası)

1. **Standartlaştırma:**
   - Girdi: `/kaggle/input/cleandata/dataset_clean-2.xlsx`
   - Çıktı: `Clean_Data_Case_DT_2025_std.xlsx`, `Clean_Data_Case_DT_2025_clean.xlsx`
2. **Temizlik + Dedup + Anomali + Konsolidasyon:**
   - Girdi: `Clean_Data_Case_DT_2025_clean.xlsx`
   - Çıktı: `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`
3. **Cinsiyet Doldurma & Normalizasyonlar:**
   - Girdi: `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`
   - Çıktı: `dataset_gender_filled.(xlsx|csv)` *(bilgi amaçlı ara çıktı)*
4. **Güvenli İmputasyon (Hasta→Grup→bilinmiyor):**
   - Girdi: `Clean_Data_Case_DT_2025_clean_final.xlsx`
   - Çıktı: `dataset_final_imputed.(xlsx|csv)`
5. **Model‑Ready Encoding + Top‑K Seçimi:**
   - Girdi: `dataset_final_imputed.xlsx`
   - Çıktı: `dataset_model_ready_best_top25.(xlsx|csv)`

> **Not:** Tüm rasgelelik içeren adımlarda `random_state / rng=42` kullanılarak tekrar edilebilirlik sağlandı.

---

## 7) Ek Notlar & Genişletme Fikirleri

- **Alias sözlüğü** (MANUAL_ALIASES) genişletilerek fuzzy’e kalmadan doğru kanoniğe çekim güçlendirilebilir.
- **Eşikler** (ör. `FUZZY_THRESHOLD=88`, `min_count=5`, `mode_ratio=0.7`) proje/kurum veri yapısına göre grid‑search benzeri bir parametrik inceleme ile optimize edilebilir.
- **“5 dk anomali”** kuralı farklı tedavi tipleri için **dinamik** eşiklere (IQR, z‑skor) genişletilebilir.
- **Özellik seçimi** için model tabanlı önem (Permutation/SHAP) raporu eklenip `TOP_K` seçimine rehberlik edebilir.
- **Saha kuralları** (ör. taraf bilgisi, “tüm vücut” önceliklendirmesi) veri sözlüğüne dökülüp otomatik testlerle güvence altına alınabilir.

---

### Dosya Özeti (Bu çalışma sırasında üretilenler)

- `Clean_Data_Case_DT_2025_std.xlsx`
- `Clean_Data_Case_DT_2025_clean.xlsx`
- `Clean_Data_Case_DT_2025_clean_final.(xlsx|csv)`
- `dataset_gender_filled.(xlsx|csv)`
- `dataset_imputed.(xlsx|csv)` / `dataset_final_imputed.(xlsx|csv)`
- `dataset_model_ready_best_top25.(xlsx|csv)`

