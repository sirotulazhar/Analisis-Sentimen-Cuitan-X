# Analisis Sentimen Cuitan X

Analisis sentimen untuk mengklasifikasikan cuitan dari platform X (Twitter) ke dalam dua kategori, yaitu Hate Speech dan Non Hate Speech. Aplikasi dibangun menggunakan Python, Machine Learning, dan Streamlit sehingga pengguna dapat melakukan prediksi melalui antarmuka web.

# Live Demo

**[Sentimen Isu Terorisme — Open App](sentimen-isu-terorisme.streamlit.app)**

## Fitur Utama

1. Prediksi Cuitan: Menganalisis satu cuitan berdasarkan teks yang dimasukkan pengguna.
2. Prediksi Massal: Mengunggah file CSV untuk menganalisis banyak cuitan sekaligus.
3. Preprocessing Teks: Membersihkan dan menormalisasi teks sebelum proses klasifikasi.
4. Hasil Klasifikasi: Menampilkan hasil prediksi dengan notifikasi merah untuk Hate Speech dan hijau untuk Non Hate Speech.
5. Unduh Hasil: Mengunduh hasil analisis massal dalam format CSV.

## Metode yang Digunakan

Analisis sentimen ini menggunakan algoritma Decision Tree untuk melakukan klasifikasi teks dengan representasi fitur TF-IDF.

## Tahapan pemrosesan teks meliputi:

1. Mengubah emoji menjadi arti dan menghapus emoji yang tersisa.
2. Mengubah angka menjadi kata dalam bahasa Indonesia.
3. Menghapus URL, mention, hashtag, dan karakter yang tidak diperlukan.
4. Mengubah teks menjadi huruf kecil serta menghapus tanda baca.
5. Menormalisasi kata alay dan slang menggunakan kamus.
6. Menghapus stopword.
7. Melakukan stemming menggunakan Sastrawi.

Hasil klasifikasi merupakan prediksi berdasarkan pola yang dipelajari model. Hasil tersebut tidak selalu mencerminkan konteks, maksud, atau penilaian manusia terhadap suatu cuitan.

## Contoh Prediksi

1. Hate Speech
   
Contoh cuitan:
@gemanawacita @MUIPusat @mohmahfudmd @YaqutCQoumas saya sangat setuju MUI itu dibubarkan biar tidak menjadi tempat berlindungnya HTI dan Teroris serta Radikalis

2. Non Hate Speech

Contoh cuitan:
@CNNIndonesia Teroris KKB ga ada rencana kemah di Kalteng nih???

## Keterangan:

1. app.py: Antarmuka aplikasi dan proses prediksi menggunakan Streamlit.
2. preprocessing.py: Fungsi pembersihan dan normalisasi teks.
3. DT.sav: Model Decision Tree yang telah disimpan.
4. vokabuler.pkl: Vocabulary yang digunakan untuk pembentukan fitur teks.
5. teroris.csv: File contoh untuk mencoba fitur prediksi CSV.
