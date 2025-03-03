# -*- coding: utf-8 -*-
"""car-price-prediction-using-various-algorithm.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14xTVyB1ZtJ14PydZnYOuLeNscRdCw3Ck

# **Car Price Prediction**

**Dataset** : https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

**Model** : Random Forest, Gradient Boosting, XGBoost, KNN, MLP

# **Data Loading**
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
import os

folder_path = "/kaggle/input/used-car-dataset-ford-and-mercedes"

csv_files = [file for file in os.listdir(folder_path)
             if file.endswith('.csv') and "unclean" not in file.lower()]

# Mapping untuk perbaikan nama brand
brand_corrections = {
    "hyundi": "hyundai",
    "merc": "mercedes",
    "cclass": "mercedes",
    "focus": "ford"
}

df_list = []
for file in csv_files:
    brand = file.split(".")[0].lower()

    brand = brand_corrections.get(brand, brand)

    df_temp = pd.read_csv(os.path.join(folder_path, file))
    df_temp = df_temp.rename(columns={"tax(£)": "tax"})
    df_temp["brand"] = brand

    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)

cols = ["brand"] + [col for col in df.columns if col != "brand"]
df = df[cols]

df.head()

"""# Exploratory Data Analysis (EDA)

## Deskripsi Variabel
"""

df.info()

df.head()

df.describe()

df.shape

"""## Univariate Analysis"""

numerical_features = [
    "year", "mileage", "price", "engineSize", "tax", "mpg"
]

categorical_features = [
    "brand", "model", "transmission", "fuelType"
]

"""### Categorical Features"""

for feature in categorical_features:
    count = df[feature].value_counts()
    percent = 100 * df[feature].value_counts(normalize=True)

    df_summary = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})
    print(f"\nDistribusi data untuk {feature}:\n", df_summary)

    plt.figure(figsize=(10, 5))
    count.plot(kind='bar', title=f'Distribusi {feature}')
    plt.xlabel(feature)
    plt.ylabel('Jumlah Sampel')
    plt.xticks(rotation=45)
    plt.show()

"""### Numerical Features"""

df.hist(bins=50, figsize=(20,15))
plt.show()

"""## Multivariate Analysis

### Categorical Features
"""

cat_features = df.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 3,  data=df, palette="Set3")
  plt.title("Rata-rata 'price (£)' Relatif terhadap - {}".format(col))

"""### Numerical Features"""

sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""# **Data Preparation**"""

df.info()

df.drop(columns=["mpg", "tax"], inplace=True)
df.head()

df.info()

# Cek jumlah missing values pada dataset
df.isnull().sum()

"""## Menangani Missing Values & Outliers"""

# Tidak ada Missing Values, lanjut atasi Outliers
# Menampilkan boxplot dari fitur-fitur numerikal untuk mendeteksi outlier
num_features = df.select_dtypes(include=['number']).columns
num_plots = len(num_features)
rows = (num_plots // 4) + (num_plots % 4 > 0)
fig, axes = plt.subplots(nrows=rows, ncols=4, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    sns.boxplot(x=df[feature], ax=axes[i])
    axes[i].set_title(feature)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Mengatasi outliers dengan metode IQR
numerical_outlier = [
    "year", "price", "mileage", "engineSize"
]

Q1 = df[numerical_outlier].quantile(0.25)
Q3 = df[numerical_outlier].quantile(0.75)
IQR = Q3 - Q1

# Menentukan batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menghapus outlier
df = df[~((df[numerical_outlier] < lower_bound) | (df[numerical_outlier] > upper_bound)).any(axis=1)]
df.shape

# Menampilkan ulang boxplot untuk mengecek apakah outlier sudah teratasi
num_features = df.select_dtypes(include=['number']).columns
num_plots = len(num_features)
rows = (num_plots // 4) + (num_plots % 4 > 0)
fig, axes = plt.subplots(nrows=rows, ncols=4, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    sns.boxplot(x=df[feature], ax=axes[i])
    axes[i].set_title(feature)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

"""## Encoding for Categorical Feature"""

# One-Hot Encoding untuk brand, transmission, fuelType
df = pd.get_dummies(df, columns=["brand", "transmission", "fuelType"], drop_first=True)

# Frequency Encoding untuk model
model_counts = df["model"].value_counts()
df["model_encoded"] = df["model"].map(model_counts)
df = df.drop(columns=["model"])

df = df.astype(int)

df.head()

"""## Dimensionality Reduction with PCA"""

from sklearn.decomposition import PCA


features_mileage_year = ["mileage", "year"]

pca_mileage_year = PCA(n_components=1, random_state=42)
df["vehicle_age_usage"] = pca_mileage_year.fit_transform(df[features_mileage_year]).flatten()

df.drop(columns=features_mileage_year, inplace=True)

# Menampilkan beberapa baris pertama hasil transformasi
df.head()

print(f"Variansi yang dijelaskan oleh komponen vehicle_age_usage: {pca_mileage_year.explained_variance_ratio_[0]:.2f}")

"""## Train-Test-Split"""

from sklearn.model_selection import train_test_split

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Jumlah data latih: {X_train.shape[0]}")
print(f"Jumlah data uji: {X_test.shape[0]}")

"""## Standarization"""

from sklearn.preprocessing import StandardScaler

numerical_features = ["engineSize", "vehicle_age_usage", "model_encoded"]

scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

df.head()

"""# Modelling"""

!pip install xgboost lightgbm

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

"""## Grid Search"""

# Model yang digunakan
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, objective="reg:squarederror"),
    "KNN": KNeighborsRegressor(),
    "LightGBM": LGBMRegressor(random_state=42)
}

# Hyperparameter tuning untuk GridSearch
param_grids = {
    "Random Forest": {
        "n_estimators": [300, 500],
        "max_depth": [10, 20],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [7, 15]
    },
    "XGBoost": {
        "n_estimators": [100, 300, 500],
        "max_depth": [7, 15],
        "learning_rate": [0.01, 0.1, 0.5]
    },
    "KNN": {
        "n_neighbors": [7, 10, 20],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "LightGBM": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.1, 0.3],
        "num_leaves": [31, 50],
        "max_depth": [-1, 15]
    }
}

best_models = {}
grid_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        grid_results[name] = {
            "MAE": -grid_search.best_score_,
            "Best Params": best_params
        }

        print(f"Best Params for {name}: {best_params}")
    else:
        best_model = model.fit(X_train, y_train)

    best_models[name] = best_model

grid_results_df = pd.DataFrame.from_dict(grid_results, orient="index")

# Urutkan berdasarkan MAE
grid_results_df = grid_results_df.sort_values(by="MAE", ascending=True)

print("\nBest Models from GridSearch (Sorted by MAE):")
print(grid_results_df)

"""# Evaluation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = {}

for name, model in best_models.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluasi pada data train
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluasi pada data test
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    results[name] = {
        "Train MAE": train_mae, "Test MAE": test_mae,
        "Train MSE": train_mse, "Test MSE": test_mse,
        "Train RMSE": train_rmse, "Test RMSE": test_rmse,
        "Train R² Score": train_r2, "Test R² Score": test_r2
    }

    print(f"{name} Performance:")
    print(f"   - Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    print(f"   - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    print(f"   - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"   - Train R² Score: {train_r2:.4f}, Test R² Score: {test_r2:.4f}")
    print("-" * 50)

# Menampilkan hasil evaluasi dalam tabel
results_df = pd.DataFrame(results).T
print("\nFinal Model Performance:")
print(results_df)

# Visualisasi hasil evaluasi (Train vs Test)
metrics = ["MAE", "MSE", "RMSE", "R² Score"]
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for i, metric in enumerate(metrics):
    train_col = f"Train {metric}"
    test_col = f"Test {metric}"

    sorted_results = results_df[[train_col, test_col]].sort_values(by=test_col, ascending=True)

    sorted_results.plot(kind="barh", ax=axes[i], color=["royalblue", "tomato"], zorder=3)
    axes[i].set_title(f"Perbandingan {metric} (Train vs Test)")
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel("Model")
    axes[i].legend(["Train", "Test"])
    axes[i].grid(zorder=0)

plt.tight_layout()
plt.show()

# Pilih beberapa sampel untuk inference
sample_indices = np.random.choice(X_test.index, size=5, replace=False)
X_sample = X_test.loc[sample_indices].copy()
y_true = y_test.loc[sample_indices]

pred_dict = {"y_true": y_true.values}

for name, model in best_models.items():
    pred_dict[f"prediksi_{name}"] = model.predict(X_sample).round(1)

df_predictions = pd.DataFrame(pred_dict, index=sample_indices)
print(df_predictions)