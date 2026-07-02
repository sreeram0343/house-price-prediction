import os
import json
import numpy as np
import pandas as pd
import time
import sys
from datetime import datetime
import sklearn
import xgboost as xgb
from src.preprocessing import preprocess_pipeline
from src.train import train_and_tune_models, save_model_artifacts
from src.evaluate import calculate_metrics, plot_and_save_results, plot_correlation_matrix
from src.logger import logger

def run_pipeline(dataset_path='data/House Price India.csv', test_size=0.2, random_state=42, cv=5, use_log_target=True):
    # Set paths
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/features.json'
    metrics_path = 'models/metrics.json'
    metadata_path = 'models/execution_metadata.json'
    
    start_time = time.time()
    run_timestamp = datetime.now().isoformat()
    
    logger.info("==================================================")
    logger.info("  Starting House Price Prediction ML Pipeline     ")
    logger.info("==================================================")
    
    # 1. Preprocessing & Data Scaling
    logger.info("[Step 1] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline(
        dataset_path, test_size=test_size, random_state=random_state, use_log_target=use_log_target
    )
    logger.info(f"Dataset successfully split and scaled.")
    logger.info(f"  Training shape: {X_train.shape}")
    logger.info(f"  Testing shape: {X_test.shape}")
    logger.info(f"  Features engineered ({len(feature_names)}): {feature_names}")
    
    # 2. Model Training and Comparison
    logger.info("[Step 2] Training and comparing regressors...")
    best_model, best_model_name, tuned_results = train_and_tune_models(X_train, y_train, cv=cv)
    
    # 3. Save Model Artifacts
    logger.info("[Step 3] Saving best model and scaler...")
    save_model_artifacts(best_model, scaler, feature_names, model_path, scaler_path, features_path)
    
    # 4. Model Evaluation
    logger.info("[Step 4] Evaluating best model on test set...")
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Compute metrics
    metrics = calculate_metrics(y_test, y_pred, is_log_scale=use_log_target)
    
    logger.info("--- Model Performance Summary (Original Price Scale) ---")
    logger.info(f"Selected Model: {best_model_name}")
    logger.info(f"R2 Score:  {metrics['r2']:.4f}")
    logger.info(f"MAE:       INR {metrics['mae']:,.2f}")
    logger.info(f"RMSE:      INR {metrics['rmse']:,.2f}")
    logger.info(f"MAPE:      {metrics['mape']:.2f}%")
    
    logger.info("--- Model Performance Summary (Log Scale) ---")
    logger.info(f"Log R2:    {metrics['log_r2']:.4f}")
    logger.info(f"Log MAE:   {metrics['log_mae']:.4f}")
    logger.info(f"Log RMSE:  {metrics['log_rmse']:.4f}")
    
    # Save metrics json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics exported to {metrics_path}")
    
    # 5. Generate and Save Visualizations
    logger.info("[Step 5] Creating performance visualizations...")
    plot_and_save_results(
        y_test, y_pred, best_model, feature_names, best_model_name, is_log_scale=use_log_target, save_dir='plots'
    )
    plot_correlation_matrix(X_train, feature_names, save_dir='plots')
    
    # Output comparison table of CV scores
    logger.info("==================================================")
    logger.info("  Model Cross-Validation Score Comparison (R2)  ")
    logger.info("==================================================")
    cv_comparison = []
    for name, data in tuned_results.items():
        cv_comparison.append({
            "Model": name,
            "CV R2 Score": data['score'],
            "Best Parameters": str(data['params'])
        })
    cv_df = pd.DataFrame(cv_comparison)
    for line in cv_df.to_string(index=False).split('\n'):
        logger.info(line)
    logger.info("==================================================")
    
    # Save metadata JSON
    execution_duration_sec = time.time() - start_time
    metadata = {
        "run_timestamp": run_timestamp,
        "execution_duration_sec": round(execution_duration_sec, 2),
        "python_version": sys.version,
        "parameters": {
            "dataset_path": dataset_path,
            "test_size": test_size,
            "random_state": random_state,
            "cv": cv,
            "use_log_target": use_log_target
        },
        "dependency_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "output_artifacts": {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "features_path": features_path,
            "metrics_path": metrics_path,
            "metadata_path": metadata_path
        },
        "selected_model": best_model_name
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Execution metadata exported to {metadata_path}")
    
    logger.info("  Pipeline Completed Successfully!                ")
    logger.info("==================================================")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="House Price Prediction ML Pipeline Orchestrator")
    parser.add_argument('--dataset', type=str, default='data/House Price India.csv', help='Path to raw dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion (0.0 to 1.0)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed value')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--no-log-target', dest='use_log_target', action='store_false', help='Do not log-transform target variable')
    parser.set_defaults(use_log_target=True)
    
    args = parser.parse_args()
    
    run_pipeline(
        dataset_path=args.dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        use_log_target=args.use_log_target
    )
