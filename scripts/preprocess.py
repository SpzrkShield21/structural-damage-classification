"""Modul untuk preprocessing data, feature engineering, dan ColumnTransformer."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Kelas untuk menambahkan fitur baru (feature engineering)."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Membuat fitur turunan baru untuk membantu model
        X['height_per_floor'] = X['height_ft_pre_eq'] / (X['count_floors_pre_eq'] + 1e-5)
        X['area_per_floor'] = X['plinth_area_sq_ft'] / (X['count_floors_pre_eq'] + 1e-5)
        return X

def get_preprocessor() -> ColumnTransformer:
    """Membuat dan mengembalikan objek ColumnTransformer untuk fitur numerik dan kategorikal.

    Returns:
        ColumnTransformer: Transformator preprocessing data.
    """
    numerical_features = [
        'count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 
        'height_ft_pre_eq', 'height_per_floor', 'area_per_floor'
    ]
    categorical_features = [
        'district_id', 'land_surface_condition', 'foundation_type', 'roof_type', 
        'ground_floor_type', 'other_floor_type', 'position', 
        'plan_configuration', 'superstructure'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor