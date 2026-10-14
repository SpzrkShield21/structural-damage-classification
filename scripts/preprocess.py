"""Modul untuk preprocessing data dan ColumnTransformer."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor() -> ColumnTransformer:
    """Membuat dan mengembalikan objek ColumnTransformer untuk fitur numerik dan kategorikal.

    Returns:
        ColumnTransformer: Transformator preprocessing data.
    """
    numerical_features = ['count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq']
    categorical_features = [
        'land_surface_condition', 'foundation_type', 'roof_type', 
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