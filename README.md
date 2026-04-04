# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri sejak tahun 2000 dan memiliki reputasi yang baik dalam mencetak lulusan berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi permasalahan serius yaitu tingginya angka siswa yang tidak menyelesaikan pendidikan (dropout). Tingginya angka dropout dapat berdampak pada menurunnya reputasi institusi, efektivitas proses pembelajaran, serta keberhasilan akademik siswa. Oleh karena itu, diperlukan pendekatan berbasis data untuk memahami faktor-faktor penyebab dropout dan membangun sistem yang mampu mendeteksi siswa berisiko sejak dini.

### Permasalahan Bisnis

* Faktor apa saja yang mempengaruhi siswa mengalami dropout.
* Bagaimana karakteristik siswa yang memiliki risiko tinggi untuk dropout.
* Bagaimana membangun model machine learning untuk memprediksi kemungkinan siswa dropout.
* Bagaimana menyediakan dashboard yang dapat membantu monitoring performa siswa secara efektif.

### Cakupan Proyek

1. Melakukan eksplorasi dan analisis data siswa (EDA).
2. Melakukan data preprocessing sebelum proses modeling.
3. Membangun model machine learning untuk prediksi dropout siswa.
4. Melakukan evaluasi performa model.
5. Mengembangkan dashboard interaktif menggunakan Streamlit.
6. Mengembangkan prototype sistem prediksi dropout berbasis Streamlit.

### Persiapan

Sumber data:
Dataset Students Performance (dataset yang disediakan oleh Dicoding: 

[student perfomance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset ini berisi informasi terkait:

* Performa akademik siswa
* Kehadiran (absensi)
* Faktor sosial dan latar belakang
* Status dropout siswa

Setup Environment:

Python version: Python 3.10

Langkah menjalankan proyek:

1. Clone repository atau download project

   ```
   git clone <link-repository-anda> cd submission
   ```
2. Membuat virtual environment

   ```
   python -m venv venv
   ```
3. Mengaktifkan virtual environment
   Windows:

   ```
   venv\Scripts\activate
   ```
4. Install dependencies

   ```
   pip install -r requirements.txt
   ```
5. Menjalankan notebook (analisis & training model)

   ```
   jupyter notebook notebook.ipynb
   ```

6. Menjalankan aplikasi Streamlit
   ```
   streamlit run app.py
   ```

## Business Dashboard

Dashboard dibuat menggunakan **Streamlit** untuk memvisualisasikan data siswa serta memonitor performa dan potensi dropout secara interaktif. Dashboard ini dirancang agar mudah digunakan oleh pihak institusi (non-teknis) dalam memahami kondisi siswa secara real-time.

Beberapa fitur utama dalam dashboard:

1. **Distribusi Dropout Siswa**
   Menampilkan perbandingan jumlah siswa dropout dan tidak dropout.
2. **Analisis Faktor Penyebab Dropout**
   Visualisasi hubungan antara nilai akademik, absensi, dan faktor lainnya terhadap dropout.
3. **Feature Importance**
   Menampilkan faktor paling berpengaruh dalam model machine learning.
4. **Filter Interaktif**
   User dapat mengeksplorasi data berdasarkan kategori tertentu.

Dashboard ini membantu institusi dalam:

* Mengidentifikasi siswa berisiko tinggi
* Memahami pola performa siswa
* Mendukung pengambilan keputusan berbasis data

🔗 Link Dashboard:

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dikembangkan menggunakan **Streamlit** yang memungkinkan pengguna untuk melakukan prediksi secara langsung.

Fitur Prototype:

* Input data siswa (nilai, absensi, dll)
* Prediksi status:
  * Berpotensi Dropout
  * Tidak Dropout
* Tampilan hasil prediksi secara real-time

Cara Menjalankan: streamlit run app.py

Cara Menggunakan:

1. Masukkan data siswa pada form input
2. Klik tombol Prediksi
3. Sistem akan menampilkan hasil prediksi dropout

## Conclusion

Berdasarkan hasil analisis data dan modeling yang telah dilakukan, diperoleh beberapa insight penting:

* Faktor utama yang mempengaruhi dropout adalah  performa akademik dan tingkat absensi siswa.
* Siswa dengan nilai rendah dan tingkat kehadiran yang rendah memiliki kemungkinan lebih tinggi untuk mengalami dropout.
* Model machine learning yang dibangun (Random Forest) mampu memberikan performa yang baik dalam memprediksi risiko dropout siswa.
* Dashboard yang dikembangkan membantu dalam memvisualisasikan kondisi siswa secara lebih mudah dan interaktif.

Dengan menggabungkan analisis data, machine learning, dan dashboard interaktif, institusi dapat melakukan deteksi dini terhadap siswa berisiko dan mengambil tindakan preventif secara lebih efektif.

### Rekomendasi Action Items

Berikut beberapa rekomendasi yang dapat dilakukan oleh Jaya Jaya Institut:

* Melakukan monitoring rutin terhadap siswa dengan nilai rendah
* Mengimplementasikan sistem early warning berbasis machine learning
* Memberikan program bimbingan atau mentoring khusus bagi siswa berisiko tinggi
* Meningkatkan pengawasan terhadap tingkat kehadiran siswa
* Menggunakan dashboard sebagai alat bantu dalam pengambilan keputusan akademik
