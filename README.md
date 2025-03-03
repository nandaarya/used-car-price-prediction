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
Pasar mobil bekas terus berkembang, tetapi harga kendaraan sering kali sulit diprediksi karena banyaknya faktor yang memengaruhi nilai jualnya. Variasi harga ini dapat disebabkan oleh perbedaan tahun produksi, jarak tempuh, kondisi kendaraan, merek, model, serta faktor eksternal seperti tren pasar.

Tanpa metode analisis yang tepat, penjual berisiko menetapkan harga terlalu rendah, sementara pembeli dapat membayar lebih dari nilai yang wajar. Oleh karena itu, machine learning dapat digunakan untuk memberikan prediksi harga yang lebih akurat dan berbasis data historis.

Dalam penelitian ini, beberapa pertanyaan utama yang akan dijawab adalah:
1. Faktor apa saja yang paling berpengaruh terhadap harga mobil bekas?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model machine learning?
3. Model machine learning apa yang memberikan hasil prediksi paling akurat?

### Goals
Untuk mengatasi permasalahan di atas, penelitian ini bertujuan untuk:
1. Mengidentifikasi faktor utama yang paling berpengaruh dalam penentuan harga mobil bekas menggunakan analisis data.
2. Melakukan preprocessing data yang mencakup pembersihan, normalisasi, encoding fitur kategorikal, serta penanganan missing values dan outliers agar model dapat bekerja dengan optimal.
3. Mengembangkan model machine learning yang akurat dan membandingkan beberapa algoritma untuk menemukan metode terbaik dalam memprediksi harga mobil bekas.

### Solution Statement
Untuk mencapai tujuan tersebut, penelitian ini mengusulkan beberapa solusi:

1. Eksplorasi dan Preprocessing Data
Mengumpulkan dataset dari sumber terpercaya.
Menganalisis data menggunakan univariate dan multivariate analysis untuk menemukan pola yang mempengaruhi harga.
Melakukan data cleaning, termasuk penanganan missing values, encoding fitur kategorikal, serta normalisasi data numerik agar model dapat memahami informasi dengan lebih baik.
2. Pemilihan Algoritma dan Optimasi Model
- Menggunakan berbagai algoritma machine learning, seperti:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - LightGBM
- Melakukan hyperparameter tuning dengan Grid Search untuk mendapatkan hasil terbaik.
3. Evaluasi Model dengan Metrik yang Tepat
Model akan dievaluasi berdasarkan beberapa metrik utama, yaitu:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Model terbaik akan ditentukan berdasarkan kecilnya kesalahan prediksi dan tingginya tingkat variansi data yang dapat dijelaskan oleh model.

Dengan pendekatan ini, model machine learning yang dikembangkan diharapkan dapat memberikan prediksi harga mobil bekas yang lebih akurat dan membantu konsumen serta pelaku industri otomotif dalam pengambilan keputusan yang lebih baik.
