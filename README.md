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

## Data Understanding
### Deskripsi Data
Dataset yang digunakan dalam penelitian ini berisi 108.540 entri data mobil bekas yang terdaftar di pasar mobil bekas di Inggris (UK). Data ini dikumpulkan melalui scraping dari berbagai listing kendaraan bekas dan telah dibersihkan untuk menghilangkan duplikasi serta memastikan keakuratan kolom yang tersedia.

Dataset ini bertujuan untuk membantu dalam memprediksi harga jual mobil bekas dengan mempertimbangkan berbagai faktor seperti merek, model, tahun produksi, jarak tempuh, tipe bahan bakar, ukuran mesin, pajak, serta efisiensi bahan bakar (miles per gallon/mpg).

### Sumber Data
Dataset dapat diakses dan diunduh melalui tautan berikut:
[[100,000 UK Used Car Data set]](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)

### Struktur Dataset
| No | Nama Kolom    | Deskripsi                                      | Tipe Data  | Jumlah Data Non-Null |
|----|-------------|--------------------------------|------------|---------------------|
| 1  | brand       | Merek mobil (misalnya, Ford, Mercedes) | object     | 108.540 |
| 2  | model       | Model spesifik dari mobil | object     | 108.540 |
| 3  | year        | Tahun produksi kendaraan | int64      | 108.540 |
| 4  | price       | Harga jual mobil dalam mata uang Pound Sterling (£) | int64      | 108.540 |
| 5  | transmission | Jenis transmisi (Manual/Automatic) | object     | 108.540 |
| 6  | mileage     | Jarak tempuh mobil dalam mil | int64      | 108.540 |
| 7  | fuelType    | Jenis bahan bakar (Petrol, Diesel, Hybrid, Electric) | object     | 108.540 |
| 8  | engineSize  | Kapasitas mesin dalam liter | float64    | 108.540 |
| 9  | tax        | Pajak kendaraan per tahun (£) | float64    | 99.187 |
| 10 | mpg        | Efisiensi bahan bakar dalam mil per galon | float64    | 99.187 |

### Analisis Kondisi Data
1. Jumlah Data: Dataset berisi 108.540 entri dengan 10 kolom fitur.

2. Kekurangan Data:
- Kolom tax dan mpg memiliki nilai null (kurang lebih 9.353 data kosong).
- Fitur lainnya memiliki data yang lengkap (tidak ada nilai null).

3. Tipe Data:
- 4 kolom kategorikal: brand, model, transmission, fuelType
- 3 kolom numerik integer: year, price, mileage
- 3 kolom numerik float: engineSize, tax, mpg

4. Ukuran Dataset: 8.3+ MB

### Exploratory Data analysis (EDA)
#### Univariate Analysis
1. Fitur Kategori
   - Brand
     
     ![Fitur Brand](https://github.com/user-attachments/assets/453a9fa0-1f1b-45e8-b369-36fa13225c0c)
     Terdapat 9 kategori pada fitur Brand, secara berurutan dari jumlahnya yang paling banyak yaitu: ford, mercedes, VW, Vauxhaul, bmw, audi, toyota, skoda, dan hyundai. Dari data persentase dapat kita simpulkan bahwa kategori dengan jumlah tertinggi adalah ford dengan 21.6 persen, kategori dengan jumlah terendah adalah hyundai dengan 4.5 persen, dan kategori tersisa memiliki persentase berkisar 5.8 - 15.7 persen.
     
   - Model
     
     ![image](https://github.com/user-attachments/assets/4cc4630f-4da0-45e8-852a-ed83c0e20408)
     Terdapat 195 kategori pada fitur Model, dengan 5 fitur dengan persentase tertinggi yaitu: Focus, C Class, Fiesta, Golf, dan Corsa. Dari persentase dapat kita simpulkan bahwa kategori dengan jumlah tertinggi adalah ford dengan 9.3 persen dan beberapa kategori hanya memiliki satu data saja.
     
   - Transmission
     
     ![image](https://github.com/user-attachments/assets/cd165698-1be0-45bb-9514-caccc196bfc6)
     Terdapat 4 kategori pada fitur Transmission, secara berurutan dari jumlahnya yang paling banyak yaitu: Manual, Semi-Auto, Automatic, dan Other. Dari data persentase dapat kita simpulkan bahwa kategori dengan jumlah tertinggi adalah Manual dengan 56.5 persen dan kategori dengan jumlah terendah adalah other dengan 0.0 persen atau 10 data saja.
     
   - Fuel Type
     
     ![image](https://github.com/user-attachments/assets/7a07d703-cae3-48a4-a798-7ae869da8ccd)
     Terdapat 5 kategori pada fitur Transmission, secara berurutan dari jumlahnya yang paling banyak yaitu: Petrol, Diesel, Hybrid, Other, dan Electric. Dari data persentase dapat kita simpulkan bahwa kategori dengan jumlah tertinggi adalah Petrol dengan 55.2 persen dan kategori dengan jumlah terendah adalah Electric dengan 0.0 persen atau 6 data saja.

2. Fitur Numerik
![image](https://github.com/user-attachments/assets/f4a565d8-c83e-40b5-9d30-e46d36922888)
  - Price
    - Peningkatan harga mobil bekas sebanding dengan penurunan jumlah sampel.
    - Rentang harga mobil bekas cukup tinggi yaitu dari skala ratusan Pound Sterling (£) hingga sekitar £60.000.
    - Setengah harga berlian bernilai di bawah £20.000.
    - Distribusi harga miring ke kanan.
      
  - Year
    - Peningkatan tahun keluaran mobil sebanding dengan peningkatan jumlah sampel.
    - Tahun keluaran mobil bekas terbaru adalah 2020.
    - Distribusi year miring ke kiri.
      
  - Mileage (Jarak Tempuh)
    - Peningkatan jarak tempuh mobil bekas sebanding dengan penurunan jumlah sampel.
    - Distribusi mileage miring ke kanan.
      
  - Engine Size
    - Tidak ada pola dalam distribusi.
    - Engine size dengan sampel terbanyak adalah 2.
      
  - Tax
    - Tidak ada pola dalam distribusi.
    - Tax dengan sampel terbanyak adalah 140.
      
  - Mpg
    - mpg memiliki distribusi yang terpusat di 50.

#### Multivariate Analysis
1. Fitur Kategori
   - Brand
     
     ![image](https://github.com/user-attachments/assets/02c58ceb-9158-4760-9951-96b5336b818b)
     Pada fitur ‘Brand’, ada perbedaan rata-rata harga. 3 brand yaitu mercedes, audi, dan bmw memiliki rata-rata harga yang lebih tinggi daripada brand lainnya. Sehingga kemungkinan fitur brand memiliki pengaruh atau dampak yang cukup besar terhadap rata-rata harga.

   - Model
     
     ![image](https://github.com/user-attachments/assets/f79f8792-6a0e-4a6d-bab2-d3f30121ff71)
     Pada fitur ‘Model’, ada banyak kategori dan tidak banyak perbedaan rata-rata harga. Hanya ada beberapa model yang memiliki perbedaan harga yang signifikan. Sehingga kemungkinan fitur brand memiliki pengaruh atau dampak yang cukup kecil terhadap rata-rata harga.

   - Transmission
     
     ![image](https://github.com/user-attachments/assets/f6296aaa-d394-4307-aeb4-525cc558df38)
     Pada fitur ‘Transmission’, ada perbedaan rata-rata harga. Terutama pada semi-auto dan manual yang memiliki selisih rata-rata yang besar. Sehingga kemungkinan fitur brand memiliki pengaruh atau dampak yang cukup besar terhadap rata-rata harga.

   - Fuel Type
  
     ![image](https://github.com/user-attachments/assets/09ccf085-3b0f-4650-a641-60d9c4d03af4)
     Pada fitur ‘Fuel Type’, ada perbedaan rata-rata harga. Terutama pada diesel dan petrol yang memiliki selisih rata-rata yang cukup besar. Sehingga kemungkinan fitur brand memiliki pengaruh atau dampak yang cukup besar meskipun tidak sebesar fitur transmission terhadap rata-rata harga.
  
2. Fitur Numerik
![image](https://github.com/user-attachments/assets/92af1a13-f5fd-4920-a3ba-a726e4e2eec6)
Dari grafik pairplot diatas, kita dapat fokus pada sumbu "price" dimana merupakan fitur target kita dan dapat menyimpulkan bahwa:
  - Fitur year memiliki korelasi positif dengan fitur price.
  - Fitur mileage memiliki korelasi negatif dengan fitur price.
  - Fitur engine size, tax, dan mpg tidak memiliki korelasi yang kuat dengan fitur price karena memiliki pola yang cukup acak.

![image](https://github.com/user-attachments/assets/6f11e5be-2403-4ba5-b2a1-680bbfe51ff5)
Untuk lebih jelasnya, kita dapat mengamati grafik korelasi diatas yang menunjukkan nilai korelasi fitur price dengan fitur numerik lainnya. Dapat disimpulkan bahwa:
  - fitur year, mileage, dan engine size memiliki korelasi yang cukup kuat dengan fitur price.
  - fitur tax dan mpg memiliki korelasi dengan fitur price, tetapi tidak cukup kuat.

## Data Preparation
Sebelum membangun model machine learning, diperlukan tahapan data preparation untuk memastikan bahwa data memiliki kualitas yang baik dan dapat meningkatkan performa model. Tahapan ini mencakup pembersihan data, transformasi fitur, encoding variabel kategorikal, reduksi dimensi, serta standarisasi fitur.

Langkah-langkah yang dilakukan dalam data preparation untuk prediksi harga mobil bekas adalah sebagai berikut:
1. Menghapus Fitur yang Memiliki Korelasi Lemah terhadap Harga
   Teknik yang digunakan: Feature Selection
   Fitur tax dan mpg dihapus dari dataset karena memiliki korelasi yang lemah terhadap harga (price) berdasarkan analisis korelasi sebelumnya.

   Alasan:
   - Menghapus fitur yang tidak signifikan terhadap target variabel dapat mengurangi kompleksitas model.
   - Fitur dengan korelasi lemah tidak memberikan kontribusi yang berarti dalam prediksi harga.

2. Menangani Missing Values dan Outliers
   Teknik yang digunakan: Metode IQR
   Dataset tidak memiliki missing values, sehingga tidak diperlukan imputasi. Namun, outliers dalam variabel harga diatasi menggunakan metode Interquartile Range (IQR) untuk mendeteksi dan menghapus data dengan nilai ekstrem.

   Alasan:
   - Outliers dapat menyebabkan model overfitting atau memberikan prediksi yang tidak akurat.
   - Metode IQR lebih robust terhadap distribusi data dibandingkan metode statistik lainnya.
     
3. Encoding untuk Fitur Kategorikal
   Teknik yang digunakan: One-Hot Encoding & Frequency Encoding
   Fitur brand, transmission, dan fuelType diencoding menggunakan One-Hot Encoding karena jumlah kategorinya tidak terlalu banyak. Sementara itu, fitur model diencoding menggunakan Frequency Encoding karena memiliki terlalu banyak kategori untuk One-Hot Encoding.

   Alasan:
   - One-Hot Encoding cocok untuk variabel dengan jumlah kategori terbatas, menghindari bias numerik.
   - Frequency Encoding lebih efisien untuk variabel dengan kategori yang sangat banyak, menghindari curse of dimensionality.
   - Untuk konversi data kategorikal ke data numerik agar dapat di proses model machine learning
     
4. Reduksi Dimensi dengan PCA
   Teknik yang digunakan: Principal Component Analysis (PCA)
   Fitur mileage dan year memiliki korelasi tinggi, sehingga dilakukan reduksi dimensi menggunakan PCA dengan menggabungkan kedua fitur tersebut.

   Alasan:
   - Mengurangi redundansi antar fitur yang memiliki hubungan kuat.
   - Menghindari multikolinearitas, yang dapat menyebabkan model menjadi kurang stabil.
   - Mengurangi fitur untuk meringankan komputasi tanpa menghilangkan informasi.
     
5. Pembagian Data (Train-Test Split)
   Data dibagi menjadi 90% data training dan 10% data testing untuk memastikan model dapat dievaluasi dengan baik menggunakan data yang tidak terlihat sebelumnya.

   Alasan:
   - Memisahkan data untuk mengevaluasi performa model dengan data baru.
   - Memastikan model tidak overfitting dengan hanya belajar dari data training.
   - Data dibagi dengan perbandingan 90:10 karena dataset berukuran besar (100.000).

6. Standarisasi Fitur Numerik
   Teknik yang digunakan: StandardScaler (Standarisasi Data)
   Fitur numerik diskalakan menggunakan StandardScaler agar memiliki distribusi normal dengan mean 0 dan standar deviasi 1.

   Alasan:
   - Meningkatkan stabilitas dan konvergensi model berbasis gradien, seperti Gradient Boosting dan XGBoost.
   - Menghindari skala yang terlalu besar pada fitur tertentu, yang dapat memengaruhi performa model.

## Modeling
## Evaluation
