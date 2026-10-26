import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load dataset using pandas."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean data and engineer physics-informed domain features."""
    df = df.copy()

    # Normalize AI4I style column names if present
    column_mapping = {
        'Type': 'machine_type',
        'Air temperature [K]': 'air_temperature',
        'Process temperature [K]': 'process_temperature',
        'Rotational speed [rpm]': 'rotational_speed',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'tool_wear',
        'Machine failure': 'failure'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Drop raw IDs or raw failure codes if present
    drop_unneeded = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in drop_unneeded:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Ensure planned & actual running times exist
    if 'planned_production_time' not in df.columns:
        df['planned_production_time'] = 480
    if 'actual_running_time' not in df.columns:
        df['actual_running_time'] = (df['planned_production_time'] - (df['tool_wear'] * 0.5)).clip(lower=300)

    # Downtime and Availability
    df['downtime'] = df['planned_production_time'] - df['actual_running_time']
    df['availability'] = df['actual_running_time'] / df['planned_production_time']

    # Physical domain feature engineering
    df['temp_diff'] = df['process_temperature'] - df['air_temperature']
    df['power'] = df['torque'] * (df['rotational_speed'] * 2 * np.pi / 60)
    df['overstrain'] = df['tool_wear'] * df['torque']
    df['temp_ratio'] = df['process_temperature'] / df['air_temperature']

    # Clean target failure column
    if 'failure' in df.columns:
        df['failure'] = df['failure'].apply(lambda x: 1 if x == 1 else 0)

    # Encode categorical machine_type
    if 'machine_type' in df.columns:
        le = LabelEncoder()
        df['machine_type_encoded'] = le.fit_transform(df['machine_type'].astype(str))

    return df
