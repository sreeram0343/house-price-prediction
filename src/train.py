import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
from src.logger import logger

def train_and_tune_models(X_train, y_train, cv=5):
    """
    Trains and compares multiple regressors using GridSearchCV.
    Returns:
        best_model: The overall best-performing model.
        best_model_name: The name of the best model.
        results: Dictionary containing all tuned models and their CV scores.
    """
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0, 100.0]}),
        'Lasso': (Lasso(), {'alpha': [0.001, 0.01, 0.1, 1.0]}),
        'Decision Tree': (
            DecisionTreeRegressor(random_state=42),
            {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'Random Forest': (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': [50, 100],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5]
            }
        ),
        'Gradient Boosting': (
            GradientBoostingRegressor(random_state=42),
            {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
        ),
        'XGBoost': (
            xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            {
                'n_estimators': [50, 100],
                'max_depth': [5, 7],
                'learning_rate': [0.05, 0.1]
            }
        )
    }

    y_train_flat = y_train.values.ravel() if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train.ravel()
    
    tuned_models = {}
    best_score = -np.inf
    best_model = None
    best_model_name = ""
    
    logger.info("--- Model Training & Tuning Started ---")
    for name, (model, param_grid) in models.items():
        logger.info(f"Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='r2',
            cv=cv,
            n_jobs=-1,
            error_score='raise'
        )
        grid_search.fit(X_train, y_train_flat)
        
        cv_score = grid_search.best_score_
        logger.info(f"  Best CV R2 Score: {cv_score:.4f}")
        logger.info(f"  Best Params: {grid_search.best_params_}")
        
        tuned_models[name] = {
            'model': grid_search.best_estimator_,
            'score': cv_score,
            'params': grid_search.best_params_
        }
        
        if cv_score > best_score:
            best_score = cv_score
            best_model = grid_search.best_estimator_
            best_model_name = name
            
    logger.info(f"Winner: {best_model_name} with CV R2: {best_score:.4f}")
    return best_model, best_model_name, tuned_models

def save_model_artifacts(model, scaler, feature_names, model_path='models/best_model.pkl', scaler_path='models/scaler.pkl', features_path='models/features.json'):
    """Serializes the model, scaler, and feature list."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save features list
    import json
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    logger.info(f"Features list saved to {features_path}")
