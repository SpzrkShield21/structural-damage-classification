"""Modul untuk melatih model Machine Learning menggunakan Pipeline."""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scripts.preprocess import get_preprocessor

def train_and_save_model(train_path: str = 'train.csv', model_output_path: str = 'models/pipeline_model.pkl') -> None:
    """Melatih model klasifikasi menggunakan Random Forest dan Pipeline, lalu menyimpannya.

    Args:
        train_path (str): Jalur ke file data latih.
        model_output_path (str): Jalur penyimpanan file model .pkl.
    """
    print("Memuat data latih...")
    df = pd.read_csv(train_path)
    
    # Mengambil subset data (misal 100.000 baris) agar proses training lebih cepat
    print("Melakukan sampling data untuk efisiensi pelatihan...")
    df_sample = df.sample(n=100000, random_state=42)
    
    X = df_sample.drop(columns=['building_id', 'district_id', 'vdcmun_id', 'ward_id', 'damage_grade'])
    y = df_sample['damage_grade']
    
    print("Menyiapkan preprocessor dan pipeline...")
    preprocessor = get_preprocessor()
    
    # Menggunakan Random Forest (Non-Neural Network sesuai aturan tugas)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("Sedang melatih model...")
    pipeline.fit(X, y)
    print("Pelatihan selesai!")
    
    # Menyimpan model ke folder models/
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Model berhasil disimpan ke {model_output_path}")

if __name__ == '__main__':
    train_and_save_model()