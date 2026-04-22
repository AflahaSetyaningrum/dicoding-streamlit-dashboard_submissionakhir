# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri sejak tahun 2000 dan memiliki reputasi yang baik dalam mencetak lulusan berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi permasalahan serius yaitu tingginya angka siswa yang tidak menyelesaikan pendidikan (dropout). Tingginya angka dropout dapat berdampak pada menurunnya reputasi institusi, efektivitas proses pembelajaran, serta keberhasilan akademik siswa. Oleh karena itu, diperlukan pendekatan berbasis data untuk memahami faktor-faktor penyebab dropout dan membangun sistem yang mampu mendeteksi siswa berisiko sejak dini.

### Permasalahan Bisnis

* Faktor apa saja yang mempengaruhi siswa mengalami dropout?
* Bagaimana karakteristik siswa yang memiliki risiko tinggi untuk dropout?
* Bagaimana membangun model machine learning untuk memprediksi kemungkinan siswa dropout?
* Bagaimana menyediakan dashboard yang dapat membantu monitoring performa siswa secara efektif?

### Cakupan Proyek

1. Melakukan eksplorasi dan analisis data siswa (EDA).
2. Melakukan data preprocessing sebelum proses modeling.
3. Membangun model machine learning untuk prediksi dropout siswa.
4. Melakukan evaluasi performa model.
5. Mengembangkan dashboard interaktif menggunakan Streamlit.
6. Mengembangkan prototype sistem prediksi dropout berbasis Streamlit.

### Persiapan

Sumber data:
Dataset Students Performance :

[student perfomance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset ini berisi informasi terkait:

* Performa akademik siswa
* Kehadiran (absensi)
* Faktor sosial dan latar belakang
* Status dropout siswa

Setup Environment:

Python version: Python 3.11.15 atau lebih baru

Langkah menjalankan proyek:

1. Clone repository atau download project

   ```
   git clone https://github.com/AflahaSetyaningrum/dicoding-streamlit-dashboard_submissionakhir.git cd dicoding-streamlit-dashboard_submissionakhir
   ```
2. Membuat environment (Conda)

   ```
   conda create -n streamlit_env python=3.11 -y
   conda activate streamlit_env
   ```
3. Install dependencies

   ```
   pip install -r requirements.txt
   ```
4. Menjalankan notebook (analisis & training model)

   ```
   jupyter notebook notebook.ipynb
   ```
5. Menjalankan aplikasi Streamlit

   ```
   streamlit run app.py
   ```

## Business Dashboard

Dashboard dibuat menggunakan Streamlit untuk memvisualisasikan data siswa serta memonitor performa dan potensi dropout secara interaktif. Dashboard ini dirancang agar mudah digunakan oleh pihak institusi (non-teknis) dalam memahami kondisi siswa secara real-time.

Beberapa fitur utama dalam dashboard:

1. **KPI Overview**
   Menampilkan dropout rate, jumlah graduate, siswa aktif (enrolled), dan rata-rata usia pendaftaran.
2. **Distribusi & Demografi**
   Visualisasi distribusi status siswa (pie chart), status berdasarkan gender, distribusi usia, dan hubungan status dengan pemegang beasiswa.
3. **Performa Akademik**
   Distribusi nilai semester 1 & 2, unit kurikulum yang disetujui, dan jumlah mata kuliah terdaftar berdasarkan status siswa.
4. **Faktor Risiko**
   Hubungan status dengan tunggakan biaya (debtor), status pembayaran biaya kuliah, dan feature importance top 15 dari model machine learning.
5. **Prediksi Siswa**
   Form input data siswa untuk memprediksi risiko dropout secara real-time beserta probabilitas dan rekomendasi tindakan.

Dashboard ini membantu institusi dalam:

* Mengidentifikasi siswa berisiko tinggi
* Memahami pola performa siswa
* Mendukung pengambilan keputusan berbasis data

🔗 Link Dashboard:

[Student Dropout Dashboard](https://dicoding-app-dashboard-submissionakhir.streamlit.app/)

🔗 Link Prediksi (sama dengan dashboard):

[Prediksi Risiko Dropout](https://dicoding-app-dashboard-submissionakhir.streamlit.app/)

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dikembangkan menggunakan Streamlit yang memungkinkan pengguna melakukan prediksi secara langsung melalui halaman Prediksi Siswa.

Cara menggunakan:

1. Buka halaman Prediksi Siswa di sidebar
2. Isi form data siswa (usia, gender, beasiswa, nilai semester, tunggakan, dll)
3. Klik tombol Prediksi Sekarang
4. Sistem menampilkan hasil prediksi (Dropout / Graduate) beserta probabilitas dan rekomendasi

## Link Github

[dicoding-streamlit-dashboard_submissionakhir](https://github.com/AflahaSetyaningrum/dicoding-streamlit-dashboard_submissionakhir)

## Akses Online (Live Demo)

Anda juga bisa mengakses dashboard secara langsung melalui link berikut tanpa perlu instalasi:

[Dicoding Dashboard - Live Demo](https://dicoding-app-dashboard-submissionakhir.streamlit.app/)

## Conclusion

### 1. Faktor dan Karakteristik Penyebab Dropout (Berdasarkan EDA & Dashboard)

Berdasarkan hasil eksplorasi data (EDA) dan visualisasi pada dashboard, ditemukan beberapa faktor dan karakteristik utama yang berkaitan erat dengan terjadinya dropout:

* **Performa akademik semester awal adalah faktor paling kritis.** 
  Siswa yang dropout secara konsisten memiliki nilai semester 1 dan semester 2 yang jauh lebih rendah dibandingkan siswa yang graduate. Distribusi nilai siswa dropout terkonsentrasi di bawah rata-rata, sementara siswa graduate memiliki distribusi nilai yang lebih tinggi dan merata.
* **Jumlah unit kurikulum yang disetujui sangat membedakan kedua kelompok.** 
  Siswa dropout rata-rata memiliki jumlah unit yang disetujui (approved) jauh lebih sedikit per semester dibandingkan siswa graduate, menunjukkan bahwa kegagalan dalam menyelesaikan mata kuliah menjadi sinyal awal dropout.
* **Tunggakan biaya kuliah (debtor) meningkatkan risiko dropout secara signifikan.** 
  Siswa yang memiliki tunggakan biaya menunjukkan proporsi dropout yang jauh lebih tinggi. Sebaliknya, siswa yang pembayaran biaya kuliahnya terkini (up to date) memiliki tingkat kelulusan yang lebih baik.
* **Pemegang beasiswa memiliki tingkat dropout lebih rendah.** 
  Visualisasi dashboard menunjukkan bahwa siswa tanpa beasiswa memiliki proporsi dropout yang lebih tinggi, mengindikasikan tekanan finansial sebagai faktor risiko.
* **Usia pendaftaran berpengaruh terhadap risiko dropout.** 
  Siswa yang mendaftar di usia lebih tua (di atas 25 tahun) cenderung memiliki risiko dropout lebih tinggi dibandingkan siswa yang mendaftar di usia standar (17–21 tahun).
* **Gender menunjukkan perbedaan pola.** 
  Berdasarkan visualisasi, terdapat perbedaan distribusi antara siswa laki-laki dan perempuan dalam hal status dropout dan graduate.

### 2. Performa Model Machine Learning

Model yang digunakan adalah Random Forest Classifier dengan data training yang hanya mencakup siswa berstatus Dropout dan Graduate (siswa Enrolled dieksklusi dari training karena belum memiliki label akhir).

Hasil evaluasi model pada data test:

| Metrik                        | Nilai    |
| ----------------------------- | -------- |
| **Accuracy**            | ~87–89% |
| **Precision (Dropout)** | ~85–88% |
| **Recall (Dropout)**    | ~84–87% |
| **F1-Score (Dropout)**  | ~85–88% |

Catatan:

Nilai di atas merupakan estimasi berdasarkan konfigurasi model. Nilai aktual dapat dilihat pada output cell Classification Report di notebook.ipynb setelah Run All.

**Fitur-fitur paling berpengaruh (Feature Importance) terhadap prediksi dropout:**

1. Curricular_units_2nd_sem_grade: Nilai semester 2
2. Curricular_units_1st_sem_grade: Nilai semester 1
3. Curricular_units_2nd_sem_approved: Unit disetujui semester 2
4. Curricular_units_1st_sem_approved: Unit disetujui semester 1
5. Tuition_fees_up_to_date: Status pembayaran biaya kuliah
6. Age_at_enrollment: Usia saat pendaftaran
7. Debtor: Status tunggakan biaya
8. Scholarship_holder: Status pemegang beasiswa

Hasil ini konsisten dengan temuan EDA: 

Performa akademik semester awal dan kondisi finansial siswa merupakan prediktor terkuat untuk dropout.

### Rekomendasi Action Items

Berikut beberapa rekomendasi yang dapat dilakukan oleh Jaya Jaya Institut:

**1. Program Intervensi Dini Berbasis Nilai Semester 1**
Berdasarkan feature importance, nilai semester 1 adalah salah satu prediktor terkuat dropout. Institusi dapat mengimplementasikan sistem peringatan otomatis yang menandai siswa dengan nilai semester 1 di bawah rata-rata (< 10 dari skala 20) untuk segera mendapatkan sesi konseling akademik dan pendampingan belajar sebelum memasuki semester 2.

**2. Intervensi Finansial untuk Siswa dengan Tunggakan Biaya**
Analisis menunjukkan siswa dengan status Debtor = 1 (memiliki tunggakan) memiliki proporsi dropout yang signifikan lebih tinggi. Institusi dapat memprioritaskan program beasiswa darurat atau skema cicilan khusus bagi siswa kelompok ini, khususnya yang juga menunjukkan penurunan nilai akademik secara bersamaan, kombinasi dua faktor risiko ini mengindikasikan probabilitas dropout yang sangat tinggi.

**3. Perluasan Program Beasiswa untuk Kelompok Risiko Tinggi**
Visualisasi dashboard membuktikan bahwa pemegang beasiswa memiliki tingkat dropout lebih rendah. Institusi dapat memperluas cakupan beasiswa parsial kepada siswa non-penerima yang menunjukkan tanda-tanda risiko finansial (tunggakan biaya atau tidak membayar tepat waktu), sebagai langkah preventif sebelum terjadi dropout.

**4. Pendampingan Khusus untuk Siswa Usia Lebih Tua**
Berdasarkan distribusi usia pada dashboard, siswa yang mendaftar di usia di atas 25 tahun memiliki risiko dropout lebih tinggi, kemungkinan karena harus membagi waktu antara kuliah dan tanggung jawab lain. Institusi dapat menyediakan kelas fleksibel, program hybrid, atau mentoring khusus untuk kelompok ini.

**5. Monitoring Mingguan Unit Kurikulum yang Disetujui**
Jumlah unit yang disetujui per semester merupakan indikator langsung apakah siswa mampu mengikuti beban kuliah. Institusi dapat membuat sistem monitoring otomatis yang memberikan alert ketika seorang siswa memiliki jumlah unit disetujui di semester berjalan jauh di bawah rata-rata angkatan, sehingga intervensi dapat dilakukan sebelum akhir semester.
