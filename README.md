Proyek _Supervised Machine Learning_ untuk memprediksi tingkat kerusakan bangunan (_damage_grade_) akibat gempa bumi besar menggunakan pendekatan klasifikasi.

## 📂 Struktur Repository

```text
structural-damage-classification/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── pipeline_model.pkl
│
├── submissions/
│   └── submission.csv
│
└── reports/
    └── figures/

```

Cara Penggunaan Repository (Installation & Usage)
Clone repository ini ke komputer lokal Anda.

Install dependencies yang diperlukan dengan menjalankan perintah berikut di terminal:

```Bash
pip install -r requirements.txt
```

Melatih Model (Training):
Jalankan skrip training untuk memproses data dan melatih model menggunakan Pipeline:

```Bash
python -m scripts.train
```

Membuat Prediksi (Prediction & Submission):
Jalankan skrip prediksi untuk menghasilkan file submission.csv di dalam folder submissions/:

```Bash
python -m scripts.predict
```

Workflow & Metodologi Analisis
Preprocessing: Menggunakan ColumnTransformer untuk standarisasi fitur numerik (StandardScaler) dan enkoding fitur kategorikal (OneHotEncoder).

Pipeline Scikit-Learn: Seluruh tahapan dari data mentah hingga prediksi digabungkan ke dalam objek Pipeline terintegrasi.

Model: Menggunakan Random Forest Classifier (non-Neural Network) sesuai dengan ketentuan tugas.

Evaluasi: Performa model dinilai berdasarkan metrik Macro F1-Score pada kompetisi Kaggle.
