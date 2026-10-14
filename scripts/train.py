"""Modul untuk melatih model Machine Learning menggunakan Pipeline dan Feature Engineering."""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scripts.preprocess import get_preprocessor, FeatureEngineer

def train_and_save_model(train_path: str = 'train.csv', model_output_path: str = 'models/pipeline_model.pkl') -> None:
    """Melatih model klasifikasi menggunakan Random Forest dengan Pipeline lengkap.

    Args:
        train_path (str): Jalur ke file data latih.
        model_output_path (str): Jalur penyimpanan file model .pkl.
    """
    print("Memuat seluruh data latih...")
    df = pd.read_csv(train_path)
    
    # Memasukkan district_id sebagai fitur penting wilayah
    X = df.drop(columns=['building_id', 'vdcmun_id', 'ward_id', 'damage_grade'])
    y = df['damage_grade']
    
    print("Menyiapkan preprocessor, feature engineer, dan pipeline...")
    preprocessor = get_preprocessor()
    
    # Menggunakan Random Forest yang dioptimalkan dengan hyperparameter lebih kuat
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=35,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ('fe', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("Sedang melatih model dengan fitur tambahan...")
    pipeline.fit(X, y)
    print("Pelatihan selesai!")
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Model berhasil disimpan ke {model_output_path}")

if __name__ == '__main__':
    train_and_save_model()