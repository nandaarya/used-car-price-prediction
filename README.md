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
     Terdapat 9 kategori pada fitur Brand, secara berurutan dari jumlahnya yang paling banyak yaitu: ford, mercedes, VW, Vauxhaul, bmw, audi, toyota, skoda, dan hyundai. Dari data persentase dapat disimpulkan bahwa kategori dengan jumlah tertinggi adalah ford dengan 21.6 persen, kategori dengan jumlah terendah adalah hyundai dengan 4.5 persen, dan kategori tersisa memiliki persentase berkisar 5.8 - 15.7 persen.
     
   - Model
     
     ![image](https://github.com/user-attachments/assets/4cc4630f-4da0-45e8-852a-ed83c0e20408)
     Terdapat 195 kategori pada fitur Model, dengan 5 fitur dengan persentase tertinggi yaitu: Focus, C Class, Fiesta, Golf, dan Corsa. Dari persentase dapat disimpulkan bahwa kategori dengan jumlah tertinggi adalah ford dengan 9.3 persen dan beberapa kategori hanya memiliki satu data saja.
     
   - Transmission
     
     ![image](https://github.com/user-attachments/assets/cd165698-1be0-45bb-9514-caccc196bfc6)
     Terdapat 4 kategori pada fitur Transmission, secara berurutan dari jumlahnya yang paling banyak yaitu: Manual, Semi-Auto, Automatic, dan Other. Dari data persentase dapat disimpulkan bahwa kategori dengan jumlah tertinggi adalah Manual dengan 56.5 persen dan kategori dengan jumlah terendah adalah other dengan 0.0 persen atau 10 data saja.
     
   - Fuel Type
     
     ![image](https://github.com/user-attachments/assets/7a07d703-cae3-48a4-a798-7ae869da8ccd)
     Terdapat 5 kategori pada fitur Transmission, secara berurutan dari jumlahnya yang paling banyak yaitu: Petrol, Diesel, Hybrid, Other, dan Electric. Dari data persentase dapat disimpulkan bahwa kategori dengan jumlah tertinggi adalah Petrol dengan 55.2 persen dan kategori dengan jumlah terendah adalah Electric dengan 0.0 persen atau 6 data saja.

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
Dari grafik pairplot diatas, jika fokus pada sumbu "price" dimana merupakan fitur target, dapat disimpulkan bahwa:
  - Fitur year memiliki korelasi positif dengan fitur price.
  - Fitur mileage memiliki korelasi negatif dengan fitur price.
  - Fitur engine size, tax, dan mpg tidak memiliki korelasi yang kuat dengan fitur price karena memiliki pola yang cukup acak.

![image](https://github.com/user-attachments/assets/6f11e5be-2403-4ba5-b2a1-680bbfe51ff5)
Untuk lebih jelasnya, dapat diamati grafik korelasi diatas yang menunjukkan nilai korelasi fitur price dengan fitur numerik lainnya. Dapat disimpulkan bahwa:
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
   
   Struktur dataset terbaru:
   ![image](https://github.com/user-attachments/assets/7009362c-3b00-4bdf-bf02-79b00e210cbc)

2. Menangani Missing Values dan Outliers
   
   Teknik yang digunakan: Metode IQR
   
   Dataset tidak memiliki missing values, sehingga tidak diperlukan imputasi. Namun, outliers dalam variabel harga diatasi menggunakan metode Interquartile Range (IQR) untuk mendeteksi dan menghapus data dengan nilai ekstrem.

   Alasan:
   - Outliers dapat menyebabkan model overfitting atau memberikan prediksi yang tidak akurat.
   - Metode IQR lebih robust terhadap distribusi data dibandingkan metode statistik lainnya.
  
   Cek Missing Values:
   ![image](https://github.com/user-attachments/assets/6eac1227-ab75-4fbb-b910-b7c19017a762)

   Dataset awal:
   ![image](https://github.com/user-attachments/assets/07234aac-3f6f-416a-818b-3404f63e1db9)

   Dataset setelah dilakukan metode IQR:
   ![image](https://github.com/user-attachments/assets/5591259b-15e2-4bfa-bdff-c0f565f2ee69)
     
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
   
   Data dibagi menjadi 90% data training dan 10% data testing untuk memastikan model dapat dievaluasi dengan baik menggunakan data yang tidak terlihat sebelumnya. Jumlah data latih: 89451. Jumlah data uji: 9940.

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

## Modelling
Setelah tahap data preparation, langkah berikutnya adalah membangun model machine learning untuk memprediksi harga mobil bekas. Dalam penelitian ini, digunakan lima algoritma machine learning yang berbeda untuk membandingkan performanya, yaitu Random Forest, Gradient Boosting, XGBoost, K-Nearest Neighbors (KNN), dan LightGBM.

Agar setiap model dapat bekerja secara optimal, dilakukan hyperparameter tuning menggunakan GridSearchCV untuk menemukan kombinasi parameter terbaik yang menghasilkan performa terbaik pada data validasi.

1. Random Forest
   
   Random Forest adalah algoritma berbasis ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi prediksi dan mengurangi overfitting.

   - Parameter yang digunakan:
     - n_estimators: 300, 500
     - max_depth: 10, 20
     - min_samples_split: 5, 10
     - min_samples_leaf: 1, 2
       
   - Kelebihan:
     - Mampu menangani data non-linear dengan baik.
     - Tidak sensitif terhadap outliers.
     - Relatif mudah untuk diinterpretasikan dibandingkan boosting methods.
       
   - Kekurangan:
     - Cenderung lebih lambat dibandingkan model boosting jika jumlah estimators terlalu besar.
     - Tidak sebaik model boosting dalam menangkap pola kompleks pada dataset besar.
       
2. Gradient Boosting
   
   Gradient Boosting adalah metode boosting yang secara bertahap membangun model berdasarkan kesalahan dari model sebelumnya.

   - Parameter yang digunakan:
     - n_estimators: 100, 300, 500
     - learning_rate: 0.01, 0.1, 0.5
     - max_depth: 7, 15
       
   - Kelebihan:
     - Lebih akurat dibandingkan Random Forest dalam dataset yang besar.
     - Mampu menangani fitur yang tidak relevan dengan lebih baik.
     - Dapat dioptimalkan lebih lanjut dengan hyperparameter tuning.
       
   - Kekurangan:
     - Lebih lambat dibandingkan Random Forest karena bersifat iteratif.
     - Memerlukan tuning parameter yang lebih kompleks.
       
3. XGBoost
   
   XGBoost adalah versi yang lebih efisien dari Gradient Boosting, yang dirancang untuk memberikan performa lebih cepat dengan optimasi berbasis GPU.

   - Parameter yang digunakan:
     - n_estimators: 100, 300, 500
     - max_depth: 7, 15
     - learning_rate: 0.01, 0.1, 0.5
       
   - Kelebihan:
     - Salah satu algoritma terbaik untuk dataset tabular.
     - Memiliki fitur built-in untuk menangani missing values.
     - Mendukung parallel processing, lebih cepat dari Gradient Boosting.
       
   - Kekurangan:
     - Lebih sulit untuk dituning dibandingkan Random Forest.
     - Bisa lebih kompleks dalam interpretasi dibandingkan model lainnya.
       
4. K-Nearest Neighbors (KNN)
   
   KNN adalah model berbasis instance learning yang menentukan prediksi berdasarkan rata-rata nilai dari tetangga terdekatnya.

   - Parameter yang digunakan:
     - n_neighbors: 7, 10, 20
     - weights: uniform, distance
     - metric: euclidean, manhattan
       
   - Kelebihan:
     - Mudah diimplementasikan dan diinterpretasikan.
     - Bekerja baik pada dataset kecil.
       
   - Kekurangan:
     - Sangat lambat untuk dataset besar karena perlu menghitung jarak ke setiap titik data.
     - Rentan terhadap noise dan outliers dalam data.
       
5. LightGBM
   
   LightGBM adalah algoritma boosting berbasis pohon yang lebih ringan dan cepat dibandingkan XGBoost, dengan teknik pembelajaran berbasis histogram.

   - Parameter yang digunakan:
     - n_estimators: 100, 300, 500
     - learning_rate: 0.01, 0.1, 0.3
     - num_leaves: 31, 50
     - max_depth: -1, 15
       
   - Kelebihan:
     - Lebih cepat dibandingkan XGBoost pada dataset besar.
     - Mampu menangani dataset dengan dimensi tinggi.
     - Tidak terlalu rentan terhadap overfitting dengan parameter yang tepat.
       
   - Kekurangan:
     - Kurang stabil dibandingkan XGBoost dalam beberapa kasus.
     - Tidak seintuitif Random Forest dalam hal interpretasi model.

### Hyperparameter Tuning dengan GridSearchCV
Agar model dapat bekerja optimal, dilakukan pencarian parameter terbaik menggunakan GridSearchCV.

Tahapan yang dilakukan:

1. Menentukan range parameter yang akan diuji untuk setiap model (parameter pada setiap model diatas).
2. Melatih model dengan GridSearchCV, menggunakan validasi silang (cross-validation = 3) untuk mencari kombinasi parameter terbaik.
3. Memilih parameter terbaik berdasarkan metrik Mean Absolute Error (MAE).

Dengan GridSearchCV, setiap model diuji dengan berbagai kombinasi parameter untuk mencari konfigurasi yang menghasilkan error terkecil.

**Hasil GridSearch**
| **Model**            | **MAE**       | **Best Parameters**  |
|----------------------|--------------|------------------------------------------------------------------|
| **XGBoost**         | **1582.60**   | `learning_rate=0.1, max_depth=7, n_estimators=500`              |
| **LightGBM**        | **1584.17**   | `learning_rate=0.1, max_depth=15, n_estimators=500, num_leaves=50` |
| **Gradient Boosting** | **1584.47**  | `learning_rate=0.1, max_depth=7, n_estimators=500`              |
| **Random Forest**   | **1627.05**   | `max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=500` |
| **KNN**             | **1672.47**   | `metric='manhattan', n_neighbors=20, weights='distance'`        |

Dari hasil GridSearch diatas, didapatkan model dengan parameter terbaik dari setiap model, dengan rincian sebagai berikut:

- XGBoost → Model dengan MAE terendah (1582.60), menunjukkan performa baik dalam prediksi harga mobil bekas.
  - Best Parameters: learning_rate=0.1, max_depth=7, n_estimators=500.
- LightGBM → Performa hampir setara dengan XGBoost dengan MAE 1584.17, sedikit lebih tinggi namun tetap sangat kompetitif.
  - Best Parameters: learning_rate=0.1, max_depth=15, n_estimators=500, num_leaves=50.
- Gradient Boosting → MAE 1584.47, sangat mirip dengan LightGBM, tetapi biasanya lebih lambat dalam pelatihan dibanding LightGBM dan XGBoost.
  - Best Parameters: learning_rate=0.1, max_depth=7, n_estimators=500.
- Random Forest → Memiliki MAE lebih tinggi (1627.05) dibanding model boosting, tetapi tetap cukup kuat dalam prediksi.
  - Best Parameters: max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=500.
- KNN → Model dengan MAE tertinggi (1672.47), menunjukkan performa kurang optimal dibanding model lain.
  - Best Parameters: metric='manhattan', n_neighbors=20, weights='distance'.

## Evaluation
Evaluasi model dilakukan untuk menilai sejauh mana model machine learning yang telah dikembangkan mampu memprediksi harga mobil bekas dengan akurasi tinggi. Pada penelitian ini, digunakan empat metrik evaluasi utama, yaitu Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R² Score (Coefficient of Determination).

Pemilihan metrik ini didasarkan pada karakteristik data regresi dan kebutuhan untuk mengukur seberapa jauh prediksi model dari nilai sebenarnya.

1. Mean Absolute Error (MAE)
   
   MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. Metrik ini memberikan gambaran seberapa besar rata-rata kesalahan prediksi model dalam satuan yang sama dengan target variabel (dalam hal ini, harga mobil bekas dalam satuan mata uang Pound Sterling (£)).

   Formula:
   
   ![image](https://github.com/user-attachments/assets/b6ab2d98-c6d0-4351-a3b7-af9a87ae318b)

   dimana:
   
   ![image](https://github.com/user-attachments/assets/71c1e665-23f9-470d-95a3-729cacb401a4)

   Interpretasi:
   - MAE yang lebih rendah berarti model lebih akurat dalam memprediksi harga mobil bekas.
   - MAE tidak memperhitungkan apakah model over-prediksi atau under-prediksi, hanya melihat besarnya error secara absolut.
   
2. Mean Squared Error (MSE)

   MSE adalah rata-rata kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya. Berbeda dengan MAE, MSE memberikan penalti lebih besar terhadap kesalahan yang besar karena nilai error dikuadratkan.

   Formula:

   ![image](https://github.com/user-attachments/assets/45e799ca-db87-415f-b8ae-a4320c021820)

   Interpretasi:
   - MSE yang lebih kecil menunjukkan model lebih akurat, karena rata-rata error dikuadratkan dan tidak memperhitungkan arah error.
   - MSE lebih sensitif terhadap outliers karena kesalahan yang besar dikuadratkan, sehingga memiliki dampak yang lebih signifikan dibandingkan MAE.
   
3. Root Mean Squared Error (RMSE)

   RMSE adalah akar kuadrat dari MSE. Metrik ini memberikan gambaran error dalam skala yang sama dengan variabel target, sehingga lebih mudah diinterpretasikan dibandingkan MSE.

   Formula:

   ![image](https://github.com/user-attachments/assets/9f694a6d-7744-489a-b233-3f45063bf30d)

   Interpretasi:
   - RMSE lebih berguna dibandingkan MSE karena skalanya lebih sesuai dengan harga mobil bekas.
   - RMSE tetap sensitif terhadap outliers, tetapi lebih mudah dibandingkan MSE dalam memahami besarnya error dalam unit yang sama dengan target variabel.

4. R² Score (Koefisien Determinasi)

   R² Score mengukur seberapa baik model menjelaskan variabilitas data target. Nilai R² berkisar antara 0 hingga 1, dengan nilai yang lebih tinggi menunjukkan model lebih baik dalam menjelaskan hubungan antara fitur dengan harga mobil bekas.

   Formula:

   ![image](https://github.com/user-attachments/assets/f16b37ac-6b70-44d5-a017-5463115756de)

   Di mana:

   ![image](https://github.com/user-attachments/assets/8364d45e-0e2c-4a6f-8d61-5dc141ea8555)

   Interpretasi:
   - R² mendekati 1 berarti model sangat baik dalam menjelaskan variasi data.
   - R² mendekati 0 berarti model tidak mampu menjelaskan variasi dalam data.
   - Jika R² negatif, berarti model lebih buruk dibandingkan dengan model baseline sederhana (misalnya, rata-rata nilai target).

### Hasil Evaluasi
#### Perbandingan Performa Model
| Model             | Train MAE | Test MAE | Train MSE     | Test MSE     | Train RMSE | Test RMSE | Train R² | Test R² |
|-------------------|----------|----------|--------------|--------------|------------|------------|----------|----------|
| Random Forest    | 1180.72  | 1599.19  | 2.6233e+06   | 4.9189e+06   | 1619.66   | 2217.86   | 0.9479   | 0.9040   |
| Gradient Boosting | 1382.73  | 1571.23  | 3.4928e+06   | 4.6646e+06   | 1868.92   | 2159.78   | 0.9306   | 0.9089   |
| XGBoost          | 1422.86  | 1570.31  | 3.7578e+06   | 4.6656e+06   | 1938.51   | 2160.00   | 0.9254   | 0.9089   |
| KNN              |  57.79   | 1576.40  | 1.0887e+05   | 5.7502e+06   |  329.96   | 2397.97   | 0.9978   | 0.8878   |
| LightGBM         | 1472.76  | 1574.52  | 4.0027e+06   | 4.6753e+06   | 2000.70   | 2162.26   | 0.9205   | 0.9087   |

#### Analisis Performa Model
1. Random Forest
   
   Train MAE = 1180.72, Test MAE = 1599.19
   
   Train R² = 0.9479, Test R² = 0.9040

   Analisis:
   - Random Forest menunjukkan performa yang baik dengan R² Score yang tinggi (0.9040).
   - Terdapat sedikit overfitting, terlihat dari selisih antara Train MAE dan Test MAE yang cukup besar.
   - RMSE pada test data cukup tinggi (2217.86), menunjukkan bahwa model masih memiliki error yang relatif besar.
     
2. Gradient Boosting

   Train MAE = 1382.73, Test MAE = 1571.23
   
   Train R² = 0.9306, Test R² = 0.9089

   Analisis:
   - Model ini memiliki generalisasi yang baik karena Train R² dan Test R² tidak terlalu berbeda.
   - MAE dan RMSE lebih kecil dibandingkan Random Forest, menunjukkan bahwa model lebih akurat dalam prediksi harga mobil bekas.
   - Model ini lebih stabil dibandingkan Random Forest dan memiliki error yang lebih kecil pada test data.
     
3. XGBoost

   Train MAE = 1422.86, Test MAE = 1570.31
   
   Train R² = 0.9254, Test R² = 0.9089

   Analisis:
   - Model ini memiliki performa yang hampir sama dengan Gradient Boosting.
   - Test MAE dan Test RMSE sangat mirip dengan Gradient Boosting, tetapi Train MAE sedikit lebih besar, menunjukkan bahwa model lebih robust dan tidak terlalu overfitting.
   - R² Score pada test data (0.9089) sangat tinggi, menandakan model mampu menjelaskan variasi harga mobil dengan baik.
     
4. K-Nearest Neighbors (KNN)

   Train MAE = 57.79, Test MAE = 1576.40
   
   Train R² = 0.9978, Test R² = 0.8878

   Analisis:
   - Model ini mengalami overfitting yang cukup parah.
   - Train MAE sangat kecil (57.79), tetapi Test MAE sangat tinggi (1576.40), menunjukkan bahwa model hanya menghafal data training tetapi gagal melakukan generalisasi pada data baru.
   - R² Score pada data test lebih rendah dibandingkan model lainnya, menandakan bahwa model ini tidak cocok untuk dataset ini.
     
5. LightGBM

   Train MAE = 1472.76, Test MAE = 1574.52
   
   Train R² = 0.9205, Test R² = 0.9087

   Analisis:
   - Performa hampir setara dengan Gradient Boosting dan XGBoost, tetapi sedikit lebih buruk dalam hal RMSE.
   - Train dan Test R² Score hampir sama, menunjukkan bahwa model memiliki generalisasi yang baik.
   - Model ini dapat menjadi alternatif jika diperlukan model yang lebih cepat dibandingkan XGBoost dan Gradient Boosting.

#### Percobaan Prediksi Model
| Index  | y_true | Random Forest | Gradient Boosting | XGBoost  | KNN     | LightGBM  |
|--------|--------|---------------|-------------------|----------|---------|-----------|
| 70431  | 27555  | 21118.6       | 21742.2          | 20837.2  | 22744.6 | 23655.9   |
| 64390  |  7891  |  9708.1       |  8806.0          |  9011.8  |  7583.4 |  9106.9   |
| 8996   | 14510  | 13669.1       | 13917.8          | 13987.3  | 14510.0 | 13646.4   |
| 51511  |  7350  |  6591.9       |  7099.8          |  6913.4  |  6785.2 |  6887.8   |
| 81826  | 20290  | 17817.5       | 19266.4          | 18115.1  | 19516.6 | 19223.9   |

1. Random Forest
   - Cenderung menghasilkan prediksi yang lebih rendah daripada harga sebenarnya, terutama pada data dengan harga tinggi.
   - Pada indeks 70431, prediksi model jauh lebih kecil (21,118.6) dibandingkan harga sebenarnya (27,555).
   - Hal ini sesuai dengan analisis sebelumnya bahwa Random Forest memiliki sedikit overfitting, dengan MAE lebih tinggi dibandingkan model boosting.

2. Gradient Boosting
   - Memiliki prediksi yang lebih dekat dengan harga sebenarnya dibandingkan Random Forest.
   - Pada indeks 81826, prediksi Gradient Boosting (19,266.4) lebih dekat dengan harga asli (20,290) dibandingkan Random Forest (17,817.5).
   - Model ini memiliki MAE yang rendah dan performa yang baik dalam generalisasi.
     
3. XGBoost
   - Prediksi paling stabil dibandingkan model lainnya, dengan error yang lebih kecil secara konsisten.
   - Pada indeks 8996, prediksi XGBoost (13,987.3) sangat mendekati harga sebenarnya (14,510), lebih akurat dibandingkan model lainnya.
   - Hal ini sesuai dengan evaluasi sebelumnya, di mana XGBoost memiliki MAE paling rendah dan generalisasi yang baik.
     
4. K-Nearest Neighbors (KNN)
   - Cenderung memberikan prediksi yang lebih ekstrem atau mendekati nilai sebenarnya secara tidak konsisten.
   - Pada indeks 8996, KNN tepat memprediksi nilai sebenarnya (14,510), tetapi pada indeks 70431, prediksi KNN (22,744.6) jauh dari harga sebenarnya (27,555).
   - Hal ini sejalan dengan analisis sebelumnya bahwa KNN mengalami overfitting, dengan perbedaan besar antara Train MAE (57.79) dan Test MAE (1576.40).
     
5. LightGBM
   - Prediksi mendekati Gradient Boosting dan XGBoost, menunjukkan model ini cukup akurat dalam menangkap pola harga.
   - Pada indeks 64390, prediksi LightGBM (9,106.9) lebih akurat dibandingkan KNN (7,583.4) tetapi sedikit lebih tinggi dibandingkan XGBoost (9,011.8).
   - Model ini terbukti memiliki generalization yang baik, tetapi masih sedikit kalah dari XGBoost dalam beberapa kasus.
     
#### Model Terbaik
| Peringkat | Model              | Alasan Peringkat |
|-----------|--------------------|------------------|
| **1**         | **XGBoost**            | **MAE terendah (1582.60), prediksi paling akurat dan stabil. Model ini memiliki generalisasi terbaik dengan R² Score 0.9089.** |
| 2         | Gradient Boosting  | Performa sangat mirip dengan XGBoost (MAE 1584.47), tetapi sedikit kurang stabil dalam beberapa prediksi dibandingkan XGBoost. |
| 3         | LightGBM           | MAE hampir setara dengan Gradient Boosting (1584.17), tetapi dalam beberapa sampel prediksi sedikit kurang akurat dibandingkan XGBoost dan Gradient Boosting. |
| 4         | Random Forest      | MAE lebih tinggi (1627.05), menunjukkan model ini masih memiliki overfitting dan kurang baik dalam menangkap pola harga dibandingkan model boosting. |
| 5         | K-Nearest Neighbors (KNN) | Overfitting parah (Train MAE: 57.79, Test MAE: 1576.40), R² Score terendah (0.8878), serta prediksi tidak konsisten. Tidak cocok untuk dataset ini. |


Kesimpulan Akhir:
- **XGBoost adalah model terbaik karena memiliki MAE paling rendah, prediksi paling stabil, serta generalisasi yang baik terhadap data test.**
- Gradient Boosting dan LightGBM berada di peringkat kedua dan ketiga, dengan performa hampir sama, tetapi sedikit lebih lemah dibandingkan XGBoost.
- Random Forest memiliki MAE yang lebih tinggi dan menunjukkan tanda-tanda overfitting.
- KNN memiliki performa terburuk, dengan overfitting ekstrem dan generalisasi yang sangat buruk.
- Dengan hasil ini, XGBoost menjadi pilihan utama untuk digunakan dalam prediksi harga mobil bekas, karena memberikan keseimbangan terbaik antara akurasi, stabilitas, dan generalisasi terhadap data baru.
