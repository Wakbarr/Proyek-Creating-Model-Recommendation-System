# Laporan Proyek Machine Learning - Akbar Widianto

## Project Overview

Sistem rekomendasi telah menjadi elemen penting dalam platform digital, khususnya di industri hiburan seperti layanan streaming film. Sistem ini membantu meningkatkan pengalaman pengguna dengan menyarankan konten yang sesuai dengan preferensi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi film Indonesia yang memberikan saran personal kepada pengguna berdasarkan data yang tersedia.

Masalah yang ingin diatasi adalah kesulitan pengguna dalam menemukan film yang sesuai dengan selera mereka di tengah meningkatnya jumlah film yang tersedia. Sistem rekomendasi yang efektif diperlukan untuk mempermudah pengguna menemukan film yang relevan dan menarik. Proyek ini menggunakan pendekatan berbasis data untuk memberikan solusi yang akurat dan bermanfaat.

Referensi yang digunakan mencakup:

* Content-Based Filtering: Wikipedia
* Collaborative Filtering: Wikipedia
* TF-IDF Vectorizer: Scikit-Learn
* Cosine Similarity: Scikit-Learn
* TensorFlow: TensorFlow

## Business Understanding

### Problem Statements

* Bagaimana cara memberikan rekomendasi film yang relevan berdasarkan genre yang disukai pengguna?
* Bagaimana cara menyediakan rekomendasi film yang dipersonalisasi berdasarkan rating dari pengguna lain dengan preferensi serupa?

### Goals

* Mengembangkan sistem rekomendasi berbasis konten (Content-Based Filtering) untuk merekomendasikan film berdasarkan kesamaan genre.
* Mengembangkan sistem rekomendasi berbasis kolaboratif (Collaborative Filtering) untuk menyarankan film berdasarkan pola rating pengguna lain.

### Solution Approach

#### Content-Based Filtering:

1. Mengubah genre film menjadi representasi numerik menggunakan TF-IDF Vectorizer.
2. Mengukur kesamaan antar-film dengan Cosine Similarity.
3. Menyediakan rekomendasi top-N film berdasarkan genre yang serupa.

#### Collaborative Filtering:

1. Menggunakan embedding untuk merepresentasikan pengguna dan film.
2. Membangun model neural network untuk memprediksi rating film yang belum ditonton pengguna.
3. Merekomendasikan film dengan prediksi rating tertinggi.

## Data Understanding

Dataset yang digunakan berasal dari Kaggle dan berisi informasi tentang film Indonesia. Dataset ini mencakup 1.272 entri unik dengan berbagai atribut seperti judul, genre, dan rating pengguna.

Variabel-variabel pada Dataset:
* movie_id: ID unik film sesuai IMDb.
* title: Judul film.
* year: Tahun rilis film.
* description: Sinopsis singkat film.
* genre: Genre film (contoh: Drama, Comedy, Horror).
* rating: Rating usia film.
* users_rating: Rata-rata rating dari pengguna (skala 1.2 hingga 9.4).
* votes: Jumlah pengguna yang memberikan rating.
* languages: Bahasa yang digunakan dalam film.
* directors: Nama sutradara film.
* actors: Daftar pemeran utama.
* runtime: Durasi film dalam menit.
* user_id: ID unik pengguna (ditambahkan untuk Collaborative Filtering).

Analisis Awal Dataset:

* Total film: 1.272 entri unik.
* Jumlah genre unik: 16.
* Rating pengguna bervariasi dari 1.2 hingga 9.4, menunjukkan distribusi penilaian yang luas.

## Data Preparation

### Menangani Missing Value

* Memeriksa nilai yang hilang pada dataset menggunakan `isnull().sum()`.
* Menghapus baris dengan missing value pada kolom penting seperti genre dan users\_rating.

### Menstandardisasi Genre

* Memastikan setiap film memiliki satu genre utama untuk mempermudah proses Content-Based Filtering.

### Menyiapkan Data untuk Pemodelan

* Menghapus duplikat berdasarkan `movie_id`.
* Membuat DataFrame baru dengan kolom `id`, `nama_film`, dan `genre` untuk digunakan dalam pemodelan.

## Modeling

### Content-Based Filtering

**Proses:**

1. Menggunakan `TfidfVectorizer` untuk mengubah genre menjadi matriks numerik.
2. Menghitung kesamaan antar-film dengan `cosine_similarity`.
3. Membuat fungsi rekomendasi yang mengembalikan top-5 film dengan genre serupa.

**Kelebihan:** Sederhana dan efektif untuk rekomendasi berdasarkan fitur konten.

**Kekurangan:** Tidak mempertimbangkan preferensi pengguna lain.

### Collaborative Filtering

**Proses:**

1. Encoding `user_id` dan `movie_id` menjadi indeks numerik.
2. Membagi data menjadi 80% pelatihan dan 20% validasi.
3. Membangun model neural network dengan embedding untuk pengguna dan film menggunakan TensorFlow.
4. Melatih model untuk memprediksi rating dan merekomendasikan film dengan rating tertinggi.

**Kelebihan:** Dapat menangkap pola preferensi pengguna secara implisit.

**Kekurangan:** Membutuhkan data rating yang cukup untuk hasil optimal.

## Evaluation

### Metrik Evaluasi

* **Content-Based Filtering:** Dievaluasi secara kualitatif berdasarkan kesamaan genre pada rekomendasi.
* **Collaborative Filtering:** Menggunakan Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi rating.
  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

### Hasil Evaluasi

* **Content-Based Filtering:** Rekomendasi yang dihasilkan konsisten dengan genre film yang dipilih.
* **Collaborative Filtering:** RMSE menurun seiring bertambahnya epoch pada data pelatihan dan validasi, menunjukkan model belajar dengan baik.
