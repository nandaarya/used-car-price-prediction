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

- **XGBoost** → Model dengan MAE terendah (1582.60), menunjukkan performa baik dalam prediksi harga mobil bekas.
  - **Best Parameters: learning_rate=0.1, max_depth=7, n_estimators=500.**
- **LightGBM** → Performa hampir setara dengan XGBoost dengan MAE 1584.17, sedikit lebih tinggi namun tetap sangat kompetitif.
  - **Best Parameters: learning_rate=0.1, max_depth=15, n_estimators=500, num_leaves=50.**
- **Gradient Boosting** → MAE 1584.47, sangat mirip dengan LightGBM, tetapi biasanya lebih lambat dalam pelatihan dibanding LightGBM dan XGBoost.
  - **Best Parameters: learning_rate=0.1, max_depth=7, n_estimators=500.**
- **Random Forest** → Memiliki MAE lebih tinggi (1627.05) dibanding model boosting, tetapi tetap cukup kuat dalam prediksi.
  - **Best Parameters: max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=500.**
- **KNN** → Model dengan MAE tertinggi (1672.47), menunjukkan performa kurang optimal dibanding model lain.
  - **Best Parameters: metric='manhattan', n_neighbors=20, weights='distance'.**
    
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

![image](https://github.com/user-attachments/assets/feb3d863-776a-4e93-b2a2-b1461af7f024)

#### Analisis Performa Model
Dari evaluasi model-model diatas, dapat disimpulkan bahwa:

- **XGBoost** adalah memiliki hasil MAE terendah (1570.31) dan R² tertinggi (0.9089), menunjukkan akurasi tinggi dan generalisasi yang baik. Perbedaan kecil antara Train MAE (1422.86) dan Test MAE (1570.31) menandakan model tidak mengalami overfitting maupun underfitting.
- **Gradient Boosting** memiliki performa hampir setara dengan XGBoost dengan MAE (1571.23) dan R² (0.9089), serta selisih Train-Test MAE yang kecil, menunjukkan model ini juga tidak mengalami overfitting atau underfitting.
- **LightGBM** memiliki MAE (1574.52) dan R² (0.9087), sedikit lebih tinggi dari XGBoost dan Gradient Boosting, tetapi masih dalam kisaran performa yang baik. Selisih Train-Test MAE yang kecil menandakan model tidak mengalami overfitting atau underfitting.
- **Random Forest** memiliki MAE lebih tinggi (1599.19) dan R² lebih rendah (0.9040), serta perbedaan Train-Test MAE cukup besar (1180.72 vs. 1599.19), menunjukkan bahwa model ini mengalami sedikit overfitting.
- **KNN** memiliki MAE tertinggi (1576.40) dan R² terendah (0.8878), dengan Train MAE yang sangat rendah (57.79) dibandingkan Test MAE (1576.40), menandakan overfitting ekstrem karena model hanya "menghafal" data training tetapi gagal melakukan generalisasi pada data test.

#### Percobaan Prediksi Model
| Index  | y_true | Random Forest | Gradient Boosting | XGBoost  | KNN     | LightGBM  |
|--------|--------|---------------|-------------------|----------|---------|-----------|
| 2692   | 27099  | 29282.1       | 29299.4          | 31850.6  | 27099.0 | 30583.1   |
| 41327  | 11110  |  9536.0       | 10263.2          |  9965.5  |  9816.6 |  9864.7   |
| 58398  |  9499  | 10431.9       | 10633.1          | 10426.0  | 11599.4 | 10582.2   |
| 9339   | 11000  | 10877.1       |  9163.8          |  8947.9  | 11000.0 |  8527.3   |
| 68980  | 26995  | 27175.9       | 27000.0          | 27356.4  | 27397.9 | 27404.3   |

Diatas adalah hasil prediksi dari model-model terhadap harga mobil bekas dan perbandingannya dengan harga aktualnya. Dari data diatas, dapat disimpulkan bahwa:

- Pada Data 1 (y_true = 27099), KNN (27099.0) paling akurat, karena prediksinya tepat sama dengan harga asli. Random Forest (29282.1) dan Gradient Boosting (29299.4) mengalami overestimasi, tetapi masih dalam batas yang dapat diterima. LightGBM (30583.1) memiliki overestimasi yang lebih besar, sementara XGBoost (31850.6) memiliki kesalahan terbesar dengan prediksi tertinggi.

- Pada Data 2 (y_true = 11110), XGBoost (9965.5) dan LightGBM (9864.7) memiliki prediksi paling dekat, meskipun sedikit underestimasi. Gradient Boosting (10263.2) juga cukup mendekati harga asli, sedangkan KNN (9816.6) dan Random Forest (9536.0) mengalami underestimasi yang lebih besar dengan Random Forest sebagai model yang paling meleset.

- Pada Data 3 (y_true = 9499), XGBoost (10426.0) dan Random Forest (10431.9) memiliki prediksi paling akurat, dengan sedikit overestimasi. LightGBM (10582.2) dan Gradient Boosting (10633.1) juga memiliki prediksi mendekati harga asli, tetapi lebih tinggi dibandingkan XGBoost. KNN (11599.4) mengalami overestimasi terbesar, menunjukkan prediksi yang terlalu jauh dari nilai sebenarnya.

- Pada Data 4 (y_true = 11000), KNN (11000.0) paling akurat, karena prediksinya tepat sama dengan harga asli. Random Forest (10877.1) sedikit underestimasi tetapi masih mendekati nilai sebenarnya, sedangkan XGBoost (8947.9), Gradient Boosting (9163.8), dan LightGBM (8527.3) mengalami underestimasi yang cukup besar, dengan LightGBM sebagai model yang paling meleset.

- Pada Data 5 (y_true = 26995), semua model memiliki prediksi yang cukup akurat dalam rentang 27000.0 - 27404.3. Gradient Boosting (27000.0) memiliki prediksi paling mendekati harga asli, diikuti oleh Random Forest (27175.9). XGBoost (27356.4) dan KNN (27397.9) sedikit overestimasi, tetapi masih dalam rentang yang wajar, sedangkan LightGBM (27404.3) memiliki overestimasi tertinggi di antara model lainnya.
     
#### Model Terbaik
Berdasarkan evaluasi metrik (MAE, MSE, RMSE, dan R² Score) serta analisis prediksi terhadap data uji, berikut adalah urutan model dari yang terbaik hingga terburuk:

1. **XGBoost** - Model terbaik dengan Test MAE terendah (1570.31) dan R² tertinggi (0.9089). Memiliki performa prediksi yang stabil di sebagian besar data dan generalisasi yang baik.
2. **Gradient Boosting** - Sangat kompetitif dengan XGBoost, dengan Test MAE (1571.23) dan R² (0.9089) yang hampir identik. Namun, dalam beberapa kasus mengalami sedikit lebih banyak kesalahan dibanding XGBoost.
3. **LightGBM** - Performa mendekati XGBoost dan Gradient Boosting dengan Test MAE (1574.52) dan R² (0.9087). Meskipun masih cukup akurat, model ini terkadang mengalami underestimasi atau overestimasi yang lebih besar dibanding dua model di atas.
4. **Random Forest** - Memiliki Test MAE lebih tinggi (1599.19) dan R² lebih rendah (0.9040) dibanding model boosting. Model ini sering mengalami underestimasi, sehingga kurang optimal untuk prediksi harga mobil bekas.
5. **KNN** - Model dengan performa terburuk, memiliki Test MAE (1576.40) dan R² terendah (0.8878). Prediksi model ini sangat bervariasi—kadang sangat akurat, tetapi sering kali mengalami kesalahan yang signifikan, terutama dalam kasus overestimasi dan underestimasi ekstrem.

#### Kesimpulan
Berdasarkan evaluasi menggunakan metrik **MAE, MSE, RMSE, dan R² Score**, model **XGBoost** terbukti menjadi yang terbaik dengan **Test MAE terendah (1570.31) dan R² tertinggi (0.9089)**, menunjukkan akurasi tinggi dan kemampuan generalisasi yang baik. **Gradient Boosting dan LightGBM** masih menjadi alternatif yang kompetitif, sementara **Random Forest** mengalami sedikit overfitting, dan **KNN** menunjukkan performa terburuk karena mengalami overfitting ekstrem dan kesalahan prediksi yang lebih besar.

Dari perspektif Business Understanding, model yang dikembangkan telah menjawab semua problem statement dengan baik:
1. Faktor utama yang mempengaruhi harga mobil bekas telah diidentifikasi melalui analisis eksplorasi data, menunjukkan bahwa tahun produksi, mileage, dan model mobil memiliki pengaruh signifikan.
2. Preprocessing data yang dilakukan, seperti handling missing values, encoding fitur kategorikal, normalisasi fitur numerik, dan reduksi dimensi, telah membantu meningkatkan kualitas dataset untuk pelatihan model.
3. Model terbaik telah ditemukan, yaitu XGBoost, yang memberikan prediksi harga mobil bekas dengan tingkat akurasi tertinggi.

Penelitian ini juga berhasil mencapai setiap goals yang ditetapkan:
1. Identifikasi faktor utama dalam penentuan harga mobil bekas telah dilakukan dengan analisis data.
2. Preprocessing dan pembersihan data telah meningkatkan efektivitas model, terbukti dari selisih yang kecil antara nilai MAE pada data training dan testing.
3. Pengembangan dan evaluasi model machine learning telah dilakukan dengan membandingkan berbagai algoritma, dan solusi terbaik telah ditemukan.
   
Dampak dari solution statement yang dirancang juga terlihat jelas dalam penelitian ini:
1. Eksplorasi dan preprocessing data berhasil menghasilkan dataset yang lebih bersih dan siap untuk pelatihan model.
2. Pemilihan algoritma dan optimasi model dengan Grid Search membantu meningkatkan performa model, terbukti dari hasil evaluasi yang menunjukkan generalisasi yang baik.
3. Evaluasi model dengan metrik yang tepat memastikan model terbaik benar-benar memberikan prediksi yang akurat dan tidak hanya menghafal data.

Dengan hasil ini, model XGBoost yang dikembangkan dapat digunakan oleh penjual dan pembeli mobil bekas untuk mendapatkan estimasi harga yang lebih akurat, membantu dalam pengambilan keputusan yang lebih baik di pasar otomotif.
