import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def engineer_features(df):
    """
    Cleans data and engineers advanced features.
    """
    df = df.copy()
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Fill any null values (just in case)
    # Get only numeric columns to avoid issues
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 1. House Age & Renovations
    # Dataset Date is between 42491 and 42734 (May 2016 - Dec 2016). So base year is 2016.
    df['house_age'] = (2016 - df['Built Year']).clip(lower=0)
    df['is_renovated'] = (df['Renovation Year'] > 0).astype(int)
    
    # Years since last modification (renovation if renovated, else built year)
    df['years_since_last_mod'] = df.apply(
        lambda r: 2016 - r['Renovation Year'] if r['Renovation Year'] > 0 else 2016 - r['Built Year'], 
        axis=1
    ).clip(lower=0)
    
    # 2. Ratios and totals
    df['total_rooms'] = df['number of bedrooms'] + df['number of bathrooms']
    df['bed_bath_ratio'] = df['number of bedrooms'] / (df['number of bathrooms'] + 0.1)
    df['living_lot_ratio'] = df['living area'] / (df['lot area'] + 0.1)
    df['has_basement'] = (df['Area of the basement'] > 0).astype(int)
    
    # 3. Development/renovation comparison features
    df['renovated_living_diff'] = df['living area'] - df['living_area_renov']
    df['renovated_lot_diff'] = df['lot area'] - df['lot_area_renov']
    
    # Drop ID and raw date columns that don't help prediction
    cols_to_drop = ['id', 'Date']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

def preprocess_pipeline(filepath, test_size=0.2, random_state=42, use_log_target=True):
    """
    Runs the full preprocessing pipeline: loading, engineering, splitting, scaling.
    """
    df = load_data(filepath)
    df_engineered = engineer_features(df)
    
    # Separate features and target
    X = df_engineered.drop(columns=['Price'])
    y = df_engineered['Price']
    
    if use_log_target:
        y = np.log1p(y)
        
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled features back to DataFrame for ease of use
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, X_train.columns.tolist(), scaler
