"""Modul untuk menghasilkan file submission dari data test."""

import os
import joblib
import pandas as pd

def generate_submission(test_path: str = 'test.csv', sample_sub_path: str = 'sample_submission.csv', model_path: str = 'models/pipeline_model.pkl', output_path: str = 'submissions/submission.csv') -> None:
    """Membuat file prediksi submission untuk Kaggle.

    Args:
        test_path (str): Jalur file data test.
        sample_sub_path (str): Jalur file sample submission.
        model_path (str): Jalur file model .pkl yang sudah dilatih.
        output_path (str): Jalur output file hasil submission CSV.
    """
    print("Memuat model dan data test...")
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    
    X_test = test_df.drop(columns=['building_id', 'district_id', 'vdcmun_id', 'ward_id'])
    
    print("Melakukan prediksi pada data test...")
    predictions = pipeline.predict(X_test)
    
    submission = sample_sub.copy()
    submission['damage_grade'] = predictions
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"File submission berhasil disimpan di {output_path} dengan total {len(submission)} baris.")

if __name__ == '__main__':
    generate_submission()