# Laporan Predictive Analytics: Model Machine Learning untuk Prediksi Harga Mobil Bekas

Untuk memenuhi submission pertama Predictive Analytic kelas Machine Learning Terapan - Dicoding Academy

---

## Domain Proyek

Harga mobil bekas dipengaruhi oleh banyak faktor seperti tahun produksi, merek, model, jarak tempuh, jenis bahan bakar, dan kondisi kendaraan. Menentukan harga mobil bekas dengan tepat sangat penting bagi penjual dan pembeli agar mendapatkan harga yang adil di pasar. Prediksi harga mobil bekas yang akurat dapat membantu pengguna dalam pengambilan keputusan yang lebih baik dalam jual beli kendaraan.

Beberapa studi telah meneliti faktor-faktor yang mempengaruhi harga mobil bekas. Studi yang dilakukan oleh Kumar & Gupta [2020] menunjukkan bahwa faktor-faktor seperti usia kendaraan, jarak tempuh, dan merek memiliki korelasi yang signifikan terhadap harga jual. Penelitian lainnya oleh Smith et al. [2021] menemukan bahwa model machine learning seperti Random Forest dan XGBoost dapat memberikan hasil prediksi harga mobil dengan tingkat akurasi yang tinggi.

Dalam proyek ini, peneliti ingin mengembangkan model machine learning yang dapat memprediksi harga mobil bekas secara akurat dengan mempertimbangkan berbagai faktor yang mempengaruhi nilai jual kendaraan.

Referensi terkait:

"Car Price Prediction Using Machine Learning" oleh Kumar & Gupta (2020)
"A Comparative Analysis of Machine Learning Models for Used Car Price Prediction" oleh Smith et al. (2021)

---

## Business Understanding
### Problem Statement
Dalam penelitian ini, terdapat beberapa permasalahan utama terkait prediksi harga mobil bekas menggunakan machine learning, antara lain:
1. Faktor (fitur) apa saja yang paling berpengaruh terhadap harga mobil bekas?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Model apa yang paling sesuai untuk memprediksi harga mobil bekas dengan tingkat akurasi yang optimal?

### Goals
Untuk mengatasi permasalahan di atas, penelitian ini bertujuan untuk:
1. Mengidentifikasi faktor-faktor yang berpengaruh terhadap harga mobil bekas.
2. Melakukan pembersihan dan preprocessing data untuk meningkatkan kualitas dataset sebelum digunakan untuk membangun model ML.
3. Mengembangkan model machine learning yang akurat untuk memprediksi harga mobil bekas.

### Solution Statement
Untuk mencapai tujuan tersebut, penelitian ini mengusulkan beberapa solusi:
1. Eksplorasi dan Preprocessing Data
- Mengumpulkan dataset dari sumber terpercaya seperti Kaggle atau dataset internal.
- Menganalisis data menggunakan teknik univariate dan multivariate analysis.
- Melakukan preprocessing data, termasuk normalisasi, encoding fitur kategorikal, serta menangani missing values dan outliers.

2. Pemilihan Algoritma dan Optimasi Model
- Menggunakan berbagai algoritma machine learning seperti Random Forest, Gradient Boosting, XGBoost, dan K-Nearest Neighbors untuk menemukan model terbaik.
- Melakukan hyperparameter tuning dengan Grid Search atau Random Search untuk meningkatkan performa model.

3. Evaluasi Model dengan Metrik yang Tepat
- Menggunakan metrik evaluasi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R² score untuk menilai kualitas prediksi model.

Dengan solusi ini, diharapkan model dapat memberikan prediksi harga mobil bekas yang lebih akurat sehingga dapat membantu konsumen dan pelaku bisnis dalam pengambilan keputusan yang lebih baik.
