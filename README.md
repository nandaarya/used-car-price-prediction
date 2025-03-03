# Laporan Predictive Analytics: Model Machine Learning untuk Prediksi Harga Mobil Bekas

Untuk memenuhi submission pertama Predictive Analytic kelas Machine Learning Terapan - Dicoding Academy

---

## Domain Proyek

### Latar Belakang
Pasar mobil bekas terus berkembang pesat seiring meningkatnya produksi kendaraan baru, yang berdampak pada meningkatnya jumlah mobil bekas di pasaran. Pada tahun 2021, produksi mobil penumpang melampaui 90 juta unit, yang secara langsung mempengaruhi dinamika harga mobil bekas (Vaneesha et al., 2024).

Penentuan harga mobil bekas menjadi tantangan karena dipengaruhi oleh berbagai faktor seperti tahun produksi, merek, model, jarak tempuh, dan kondisi kendaraan. Metode tradisional seperti perbandingan manual atau penilaian subjektif oleh dealer sering kali tidak konsisten dan kurang akurat, menyebabkan harga yang tidak sesuai dengan nilai pasar (Vaneesha et al., 2024).

### Mengapa Masalah Ini Harus Diselesaikan?
Ketidakakuratan dalam estimasi harga dapat merugikan baik penjual maupun pembeli. Penjual berisiko menetapkan harga terlalu rendah, sementara pembeli bisa membayar lebih dari nilai wajar. Selain itu, dealer dan platform jual-beli membutuhkan sistem prediksi harga yang lebih transparan dan objektif untuk meningkatkan efisiensi bisnis mereka (Vaneesha et al., 2024).

Untuk mengatasi permasalahan ini, diperlukan sebuah model machine learning yang dapat memprediksi harga mobil bekas secara akurat. Model ini akan dilatih menggunakan data historis kendaraan, seperti spesifikasi mobil, usia, dan kondisi penggunaan, guna menghasilkan estimasi harga yang lebih objektif dan berdasarkan pola yang terdeteksi dari data. Penelitian sebelumnya menunjukkan bahwa metode berbasis machine learning dapat meningkatkan akurasi prediksi harga mobil bekas. Studi oleh Samruddhi dan Kumar (2020) menemukan bahwa model berbasis data mampu meningkatkan akurasi estimasi harga hingga 85%, sementara penelitian oleh Vaneesha et al. (2024) menunjukkan bahwa metode otomatisasi berbasis machine learning dapat mengurangi kesalahan prediksi secara signifikan.

Dengan membangun model prediksi berbasis machine learning, diharapkan dapat tercipta sistem yang lebih transparan dan efisien dalam menentukan harga mobil bekas, membantu pembeli, penjual, serta pelaku industri otomotif dalam pengambilan keputusan yang lebih baik.

### Referensi
- Samruddhi, K., & Kumar, R. A. (2020). Used Car Price Prediction using K-Nearest Neighbor Based Model. International Journal of Innovative Research in Applied Sciences and Engineering, 4(3), 686-689. [[Link]](https://www.ijirase.com/assets/paper/issue_1/volume_4/V4-Issue-3-686-689.pdf)
- Vaneesha, K. H., Srinivas, V., Abhishek, V., & Srinivas, S. (2024). Comparative Analysis of Machine Learning Algorithms for Used Car Price Prediction. International Journal of Current Science Research and Review, 7(9), 7220-7228. [[Link]](https://ijcsrr.org/wp-content/uploads/2024/09/39-1909-2024.pdf)

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
