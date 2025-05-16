# Laporan Proyek Machine Learning - Akbar Widianto

## Project Overview

Sistem rekomendasi telah menjadi elemen krusial dalam platform digital, terutama di industri hiburan seperti layanan streaming film. Menurut Statista, penggunaan layanan streaming di Indonesia meningkat pesat, dengan lebih dari 50 juta pengguna aktif pada tahun 2023. Sistem rekomendasi membantu meningkatkan pengalaman pengguna dengan menyarankan konten yang sesuai dengan preferensi mereka, sehingga dapat meningkatkan retensi pengguna dan pendapatan platform.

Proyek ini bertujuan untuk membangun sistem rekomendasi film Indonesia yang memberikan saran personal kepada pengguna berdasarkan data yang tersedia. Masalah yang ingin diatasi adalah kesulitan pengguna dalam menemukan film yang sesuai dengan selera mereka di tengah meningkatnya jumlah konten film. Dengan dataset yang mencakup lebih dari 1.200 film, sistem rekomendasi yang efektif sangat diperlukan untuk mempermudah pengguna menemukan film yang relevan dan menarik.

**Referensi yang Digunakan:**

* Content-Based Filtering
* Collaborative Filtering
* TF-IDF Vectorizer
* Cosine Similarity
* TensorFlow

---

## Business Understanding

### Problem Statements

Berdasarkan analisis awal, pengguna sering kali kesulitan menemukan film yang sesuai dengan selera mereka karena volume konten yang besar. Hal ini menyebabkan pengguna menghabiskan banyak waktu untuk mencari film yang relevan, yang pada akhirnya dapat mengurangi kepuasan pengguna.

* **Pernyataan Masalah 1:** Bagaimana cara memberikan rekomendasi film yang relevan berdasarkan genre yang disukai pengguna?
* **Pernyataan Masalah 2:** Bagaimana cara menyediakan rekomendasi film yang dipersonalisasi berdasarkan rating dari pengguna lain dengan preferensi serupa?

### Goals

* Mengembangkan sistem rekomendasi berbasis konten (Content-Based Filtering) untuk merekomendasikan film berdasarkan kesamaan genre.
* Mengembangkan sistem rekomendasi berbasis kolaboratif (Collaborative Filtering) untuk menyarankan film berdasarkan pola rating pengguna lain.

### Solution Approach

**Content-Based Filtering**

* Pendekatan ini dipilih karena kemampuannya untuk merekomendasikan film berdasarkan fitur konten (genre) yang serupa dengan film yang disukai pengguna. Ini cocok untuk dataset dengan informasi genre yang jelas.

**Collaborative Filtering**

* Pendekatan ini memanfaatkan pola rating dari pengguna lain untuk memberikan rekomendasi yang dipersonalisasi. Ini efektif untuk menangkap preferensi implisit pengguna berdasarkan perilaku serupa dari pengguna lain.

---

## Data Understanding

Dataset yang digunakan berasal dari Kaggle dan berisi informasi tentang film Indonesia. Dataset ini mencakup **1.272 entri unik** dengan berbagai atribut seperti judul, genre, dan rating pengguna.

**Variabel pada Dataset**

* `movie_id`: ID unik film sesuai IMDb.
* `title`: Judul film.
* `year`: Tahun rilis film.
* `description`: Sinopsis singkat film.
* `genre`: Genre film (contoh: Drama, Comedy, Horror).
* `rating`: Rating usia film.
* `users_rating`: Rata-rata rating dari pengguna (skala 1.2 hingga 9.4).
* `votes`: Jumlah pengguna yang memberikan rating.
* `languages`: Bahasa yang digunakan dalam film.
* `directors`: Nama sutradara film.
* `actors`: Daftar pemeran utama.
* `runtime`: Durasi film dalam menit.
* `user_id`: ID unik pengguna (ditambahkan untuk Collaborative Filtering).

**Analisis Awal Dataset**

* Jumlah Data: 1.272 film unik.
* Kondisi Data: Terdapat missing value pada kolom `description` (432), `genre` (36), `rating` (896), dan `runtime` (403).
* Insight dari Exploratory Data Analysis:

  * Distribusi genre menunjukkan bahwa Drama dan Comedy adalah genre yang paling umum.
  * Rating pengguna bervariasi dari 1.2 hingga 9.4, dengan rata-rata sekitar 6.1.

> *Catatan: Visualisasi distribusi genre dan rating dapat dilihat pada notebook proyek.*

---

## Data Preparation

**Teknik Data Preparation**

1. **Menangani Missing Value**

   * Memeriksa nilai yang hilang menggunakan `isnull().sum()`.
   * Menghapus baris dengan missing value pada kolom penting seperti `genre` dan `users_rating`.
     *Alasan:* Kolom ini krusial untuk pemodelan dan tidak dapat diimputasi secara akurat tanpa data tambahan.

2. **Menstandardisasi Genre**

   * Memastikan setiap film memiliki satu genre utama.
     *Alasan:* Memudahkan proses Content-Based Filtering yang bergantung pada kesamaan genre.

3. **Menyiapkan Data untuk Pemodelan**

   * Menghapus duplikat berdasarkan `movie_id`.
   * Membuat DataFrame baru dengan kolom `id`, `nama_film`, dan `genre`.
     *Alasan:* Menyederhanakan data untuk keperluan pemodelan agar lebih efisien.

**Contoh Code Snippet**

```python
# Menghapus baris dengan missing value
movies_clean = data_film.dropna()

# Membuat DataFrame baru
data_final = pd.DataFrame({
    'id': data_prepared['movie_id'].tolist(),
    'nama_film': data_prepared['title'].tolist(),
    'genre': data_prepared['genre'].tolist()
})
```

---

## Modeling and Result

### Content-Based Filtering

**Proses:**

* Menggunakan `TfidfVectorizer` untuk mengubah genre menjadi matriks numerik.
* Menghitung kesamaan antar-film dengan `cosine_similarity`.
* Membuat fungsi rekomendasi yang mengembalikan top-5 film dengan genre serupa.

**Kelebihan:** Sederhana dan efektif untuk rekomendasi berdasarkan fitur konten.
**Kekurangan:** Tidak mempertimbangkan preferensi pengguna lain.

**Contoh Output:**
Untuk film "MeloDylan" (Drama), rekomendasi yang dihasilkan:

* Hanum & Rangga: Faith & The City (Drama)
* Dear Nathan (Drama)
* Labuan Hati (Drama)
* Mata Batin (Drama)
* Love for Sale 2 (Drama)

### Collaborative Filtering

**Proses:**

* Encoding `user_id` dan `movie_id` menjadi indeks numerik.
* Membagi data menjadi 80% pelatihan dan 20% validasi.
* Membangun model neural network dengan embedding untuk pengguna dan film menggunakan TensorFlow.
* Melatih model untuk memprediksi rating dan merekomendasikan film dengan rating tertinggi.

**Kelebihan:** Dapat menangkap pola preferensi pengguna secara implisit.
**Kekurangan:** Membutuhkan data rating yang cukup untuk hasil optimal.

**Contoh Output:**
Untuk pengguna U950, rekomendasi yang dihasilkan:

* Mendadak Kaya (Comedy)
* Dilan 1991 (Drama)
* Koboy Kampus (Comedy)
* Dignitate (Drama)
* Target (Comedy)
* Surat Cinta Untuk Starla the Movie (Drama)
* Ratu Ilmu Hitam (Horror)
* Wiro Sableng 212 (Action)

---

## Evaluation

### Metrik Evaluasi

**Content-Based Filtering:**

* Metrik: Evaluasi kualitatif berdasarkan kesamaan genre pada rekomendasi.
* Penjelasan: Memastikan rekomendasi sesuai dengan genre film yang dipilih pengguna.

**Collaborative Filtering:**

* Metrik: Root Mean Squared Error (RMSE)
  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
* Penjelasan: RMSE mengukur rata-rata kesalahan kuadrat antara rating yang diprediksi dan rating sebenarnya. Nilai yang lebih rendah menunjukkan prediksi yang lebih akurat.

### Hasil Evaluasi

* **Content-Based Filtering:** Rekomendasi yang dihasilkan konsisten dengan genre film yang dipilih, menunjukkan model berhasil mengidentifikasi film dengan karakteristik serupa.
* **Collaborative Filtering:** RMSE menurun seiring bertambahnya epoch pada data pelatihan dan validasi, menunjukkan model belajar dengan baik dan mampu memprediksi rating dengan akurasi yang memadai.

![image](https://github.com/user-attachments/assets/fcd53938-5839-46de-a642-d6c27e7a3c4a)

