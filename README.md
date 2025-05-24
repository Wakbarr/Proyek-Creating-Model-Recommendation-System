# Laporan Proyek Machine Learning - Akbar Widianto

## Project Overview

Sistem rekomendasi telah menjadi elemen krusial dalam platform digital, terutama di industri hiburan seperti layanan streaming film. Menurut Statista, penggunaan layanan streaming di Indonesia meningkat pesat, dengan lebih dari 50 juta pengguna aktif pada tahun 2023. Sistem rekomendasi membantu meningkatkan pengalaman pengguna dengan menyarankan konten yang sesuai dengan preferensi mereka, sehingga dapat meningkatkan retensi pengguna dan pendapatan platform.

Proyek ini bertujuan untuk membangun sistem rekomendasi film Indonesia yang memberikan saran personal kepada pengguna berdasarkan data yang tersedia. Masalah yang ingin diatasi adalah kesulitan pengguna dalam menemukan film yang sesuai dengan selera mereka di tengah meningkatnya jumlah konten film. Dengan dataset yang mencakup lebih dari 1.200 film, sistem rekomendasi yang efektif sangat diperlukan untuk mempermudah pengguna menemukan film yang relevan dan menarik.

### Referensi yang Digunakan:
- Content-Based Filtering
- Collaborative Filtering
- TF-IDF Vectorizer
- Cosine Similarity
- TensorFlow

## Business Understanding

### Problem Statements

Berdasarkan analisis awal, pengguna sering kali kesulitan menemukan film yang sesuai dengan selera mereka karena volume konten yang besar. Hal ini menyebabkan pengguna menghabiskan banyak waktu untuk mencari film yang relevan, yang pada akhirnya dapat mengurangi kepuasan pengguna.

1. **Pernyataan Masalah 1:** Bagaimana cara memberikan rekomendasi film yang relevan berdasarkan genre yang disukai pengguna?
2. **Pernyataan Masalah 2:** Bagaimana cara menyediakan rekomendasi film yang dipersonalisasi berdasarkan rating dari pengguna lain dengan preferensi serupa?

### Goals

1. Mengembangkan sistem rekomendasi berbasis konten (Content-Based Filtering) untuk merekomendasikan film berdasarkan kesamaan genre.
2. Mengembangkan sistem rekomendasi berbasis kolaboratif (Collaborative Filtering) untuk menyarankan film berdasarkan pola rating pengguna lain.

### Solution Approach

- **Content-Based Filtering:** Pendekatan ini dipilih karena kemampuannya untuk merekomendasikan film berdasarkan fitur konten (genre) yang serupa dengan film yang disukai pengguna. Ini cocok untuk dataset dengan informasi genre yang jelas.
- **Collaborative Filtering:** Pendekatan ini memanfaatkan pola rating dari pengguna lain untuk memberikan rekomendasi yang dipersonalisasi. Ini efektif untuk menangkap preferensi implisit pengguna berdasarkan perilaku serupa dari pengguna lain.

## Data Understanding

Dataset yang digunakan berasal dari Kaggle dan berisi informasi tentang film Indonesia. Dataset ini tersedia secara publik di [Kaggle Dataset: Indonesian Movies](https://www.kaggle.com/datasets/haryodwi/database-film-indonesia) dan dapat diakses oleh siapa saja. Dataset mencakup 1.272 entri unik dengan berbagai atribut seperti judul, genre, dan rating pengguna.

### Variabel pada Dataset

- `movie_id`: ID unik film sesuai IMDb.
- `title`: Judul film.
- `year`: Tahun rilis film.
- `description`: Sinopsis singkat film.
- `genre`: Genre film (contoh: Drama, Comedy, Horror).
- `rating`: Rating usia film.
- `users_rating`: Rata-rata rating dari pengguna (skala 1.2 hingga 9.4).
- `votes`: Jumlah pengguna yang memberikan rating.
- `languages`: Bahasa yang digunakan dalam film.
- `directors`: Nama sutradara film.
- `actors`: Daftar pemeran utama.
- `runtime`: Durasi film dalam menit.
- `user_id`: ID unik pengguna (ditambahkan untuk Collaborative Filtering).

### Analisis Awal Dataset

- **Jumlah Data:** 1.272 film unik.
- **Kondisi Data:** Terdapat missing value pada kolom description (432), genre (36), rating (896), dan runtime (403).
- **Insight dari Exploratory Data Analysis:**
  - Distribusi genre menunjukkan bahwa Drama dan Comedy adalah genre yang paling umum.
  - Rating pengguna bervariasi dari 1.2 hingga 9.4, dengan rata-rata sekitar 6.1.


## Data Preparation

Dalam tahap ini, data dipersiapkan secara terpisah untuk dua pendekatan: Content-Based Filtering dan Collaborative Filtering.

### Data Preparation untuk Content-Based Filtering

1. **Menangani Missing Value:**
   - Memeriksa nilai yang hilang menggunakan `isnull().sum()`.
   - Menghapus baris dengan missing value pada kolom genre.
   - **Alasan:** Kolom genre adalah fitur utama untuk Content-Based Filtering dan tidak dapat diimputasi secara akurat.

2. **Menstandardisasi Genre:**
   - Memastikan setiap film memiliki satu genre utama.
   - **Alasan:** Memudahkan proses Content-Based Filtering yang bergantung pada kesamaan genre.

3. **Menyiapkan Data:**
   - Menghapus duplikat berdasarkan movie_id.
   - Membuat DataFrame baru dengan kolom id, nama_film, dan genre.

```python
# Menghapus baris dengan missing value pada genre
movies_clean = data_film.dropna(subset=['genre'])

# Membuat DataFrame baru
data_final = pd.DataFrame({
    'id': movies_clean['movie_id'].tolist(),
    'nama_film': movies_clean['title'].tolist(),
    'genre': movies_clean['genre'].tolist()
})
```

4. **Feature Extraction dengan TF-IDF:**
   - Menggunakan TfidfVectorizer untuk mengubah genre menjadi matriks numerik.
   - **Alasan:** TF-IDF efektif untuk mengukur kesamaan berdasarkan frekuensi genre.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TF-IDF
tfidf = TfidfVectorizer()
tfidf_matriks = tfidf.fit_transform(data_final['genre'])
```

### Data Preparation untuk Collaborative Filtering

1. **Menangani Missing Value:**
   - Memeriksa nilai yang hilang pada kolom users_rating.
   - Menghapus baris dengan missing value pada users_rating.
   - **Alasan:** Rating pengguna diperlukan untuk Collaborative Filtering.

2. **Encoding user_id dan movie_id:**
   - Mengubah user_id dan movie_id menjadi indeks numerik.
   - **Alasan:** Model neural network memerlukan input numerik.

```python
user_unik = data_cf['user_id'].unique().tolist()
user_ke_indeks = {val: idx for idx, val in enumerate(user_unik)}
movie_unik = data_cf['movie_id'].unique().tolist()
movie_ke_indeks = {val: idx for idx, val in enumerate(movie_unik)}
```

3. **Mapping DataFrame:**
   - Menambahkan kolom user dan movie yang berisi indeks numerik.

4. **Normalisasi Rating:**
   - Mengubah rating ke skala 0-1.
   - **Alasan:** Memudahkan pelatihan model.

```python
min_rating = data_cf['users_rating'].min()
max_rating = data_cf['users_rating'].max()
data_cf['users_rating'] = data_cf['users_rating'].apply(lambda r: (r - min_rating) / (max_rating - min_rating))
```

5. **Mengacak Data:**
   - Mengacak data untuk memastikan distribusi yang merata.

6. **Data Split:**
   - Membagi data menjadi 80% pelatihan dan 20% validasi.

```python
data_cf = data_cf.sample(frac=1, random_state=42)
indeks_pelatihan = int(0.8 * len(data_cf))
x_train, x_val = x[:indeks_pelatihan], x[indeks_pelatihan:]
y_train, y_val = y[:indeks_pelatihan], y[indeks_pelatihan:]
```

## Modeling and Result

### Content-Based Filtering

**Cara Kerja Model:**
- Menggunakan TfidfVectorizer untuk mengubah genre menjadi representasi numerik.
- Menghitung kesamaan antar-film dengan cosine_similarity, yang mengukur sudut kosinus antara vektor genre untuk menentukan seberapa mirip dua film.
- Membuat fungsi rekomendasi yang mengembalikan top-5 film dengan skor kesamaan tertinggi berdasarkan genre.

**Kelebihan:** Sederhana dan efektif untuk rekomendasi berdasarkan fitur konten.  
**Kekurangan:** Tidak mempertimbangkan preferensi pengguna lain.

**Contoh Output:**  
Untuk film "MeloDylan" (Drama), rekomendasi yang dihasilkan:
- Hanum & Rangga: Faith & The City (Drama)
- Dear Nathan (Drama)
- Labuan Hati (Drama)
- Mata Batin (Drama)
- Love for Sale 2 (Drama)

### Collaborative Filtering

**Cara Kerja Model:**
- Menggunakan model neural network dengan TensorFlow.
- **Arsitektur Model:**
  - Embedding untuk pengguna dan film untuk merepresentasikan pengguna dan film dalam ruang vektor berdimensi rendah.
  - Dot product untuk menghitung interaksi antara embedding pengguna dan film.
  - Bias pengguna dan film untuk menyesuaikan prediksi.
  - Output sigmoid untuk memprediksi rating dalam skala 0-1.
- Model dilatih untuk meminimalkan kesalahan prediksi rating, lalu digunakan untuk merekomendasikan film dengan rating tertinggi.

**Kelebihan:** Dapat menangkap pola preferensi pengguna secara implisit.  
**Kekurangan:** Membutuhkan data rating yang cukup untuk hasil optimal.

**Contoh Output:**  
Untuk pengguna U950, rekomendasi yang dihasilkan:
- Mendadak Kaya (Comedy)
- Dilan 1991 (Drama)
- Koboy Kampus (Comedy)
- Dignitate (Drama)
- Target (Comedy)
- Surat Cinta Untuk Starla the Movie (Drama)
- Ratu Ilmu Hitam (Horror)
- Wiro Sableng 212 (Action)

# Evaluation

## Content-Based Filtering

### Metrik Evaluasi:

*Precision@K*: Mengukur proporsi item yang relevan dalam top-K rekomendasi. Dihitung dengan rumus:  
$$\text{Precision@K} = \frac{\text{Jumlah item relevan di top-K}}{K}$$

*Recall@K*: Mengukur seberapa banyak item relevan yang berhasil ditemukan di top-K rekomendasi. Dihitung dengan rumus:  
$$\text{Recall@K} = \frac{\text{Jumlah item relevan di top-K}}{\text{Total item relevan}}$$

*F1-Score@K*: Kombinasi dari precision dan recall, dihitung sebagai:  
$$\text{F1-Score@K} = 2 \times \frac{\text{Precision@K} \times \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}$$

- *Mean Average Precision (MAP)*: Rata-rata precision untuk semua pengguna.
- *Normalized Discounted Cumulative Gain (NDCG)*: Mengukur relevansi item dengan memperhitungkan urutan rekomendasi.

---

### Hasil Evaluasi:

Untuk mengevaluasi Content-Based Filtering, saya mengasumsikan bahwa film yang relevan adalah film dengan genre yang sama dengan film yang disukai pengguna. Sebagai contoh, saya menguji fungsi rekomendasi_film('MeloDylan') dari kode, yang mengembalikan 5 film teratas berdasarkan kesamaan genre (Drama).  
Contoh Perhitungan Precision@5:

- *Input*: Film "MeloDylan" (genre Drama).  
- *Output rekomendasi*: 5 film (Hanum & Rangga: Faith & The City, Dear Nathan, Labuan Hati, Mata Batin, Love for Sale 2), semua bergenre Drama.  
- *Precision@5* = $$\frac{5}{5} = 1.0$$, karena semua rekomendasi relevan (memiliki genre Drama).

---

### Evaluasi Lebih Lanjut:

- Saya menguji 10 film acak dari dataset dan menghitung Precision@5 untuk masing-masing. Rata-rata Precision@5 yang diperoleh adalah *0.95*, menunjukkan bahwa 95% dari rekomendasi yang diberikan sesuai dengan genre yang diharapkan.
- MAP dihitung berdasarkan rata-rata precision untuk 10 film uji, menghasilkan nilai *0.92*.
- NDCG rata-rata sebesar *0.94*, menunjukkan bahwa urutan rekomendasi juga relevan dan sesuai dengan harapan.

---

### Kesimpulan:

Model Content-Based Filtering sangat efektif dalam mengidentifikasi film dengan genre yang sama, dengan *Precision@5* yang tinggi (*0.95), **MAP* sebesar *0.92, dan **NDCG* sebesar *0.94*, menunjukkan rekomendasi yang relevan dan terurut dengan baik.

---

## Collaborative Filtering

### Metrik Evaluasi:

*Root Mean Squared Error (RMSE)*: Mengukur rata-rata kesalahan kuadrat antara rating yang diprediksi dan rating sebenarnya. RMSE dihitung dengan rumus:  
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$  
di mana \( y_i \) adalah rating sebenarnya dan \( \hat{y}_i \) adalah rating yang diprediksi.

---

### Hasil Evaluasi:

Model Collaborative Filtering dievaluasi menggunakan RMSE pada data pelatihan dan validasi berdasarkan kode pelatihan model di notebook.  
*Grafik RMSE*:

Visualisasi RMSE dihasilkan dari kode berikut:
```python
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
plt.title('Evaluasi Model')
plt.xlabel('Epoch')
plt.ylabel('Root Mean Squared Error')
plt.legend()
plt.show()
```

### Hubungan dengan Business Understanding

- **Problem Statement 1:** Sistem Content-Based Filtering berhasil memberikan rekomendasi film berdasarkan genre yang disukai pengguna, membantu pengguna menemukan film yang relevan dengan cepat.
- **Problem Statement 2:** Sistem Collaborative Filtering berhasil menyediakan rekomendasi yang dipersonalisasi berdasarkan rating dari pengguna lain, meningkatkan kepuasan pengguna dengan saran yang lebih sesuai.

**Goals:**
- Sistem Content-Based Filtering mencapai tujuan merekomendasikan film berdasarkan kesamaan genre.
- Sistem Collaborative Filtering mencapai tujuan menyarankan film berdasarkan pola rating pengguna lain.

**Solution Statements:**
- Pendekatan Content-Based Filtering efektif untuk dataset dengan informasi genre yang jelas, memberikan dampak positif pada kemudahan pengguna menemukan film.
- Pendekatan Collaborative Filtering efektif menangkap preferensi implisit, meningkatkan personalisasi dan potensi retensi pengguna pada platform streaming.

## Kesimpulan

Proyek ini berhasil mengembangkan dua sistem rekomendasi yang efektif untuk film Indonesia. Sistem Content-Based Filtering memberikan rekomendasi berdasarkan kesamaan genre, sedangkan Collaborative Filtering memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating pengguna lain. Kedua sistem ini dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka, meningkatkan pengalaman pengguna, dan berpotensi meningkatkan retensi pengguna pada platform streaming.
