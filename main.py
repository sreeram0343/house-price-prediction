import os
import json
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.train import train_and_tune_models, save_model_artifacts
from src.evaluate import calculate_metrics, plot_and_save_results

def run_pipeline():
    # Set paths
    dataset_path = 'data/House Price India.csv'
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    features_path = 'models/features.json'
    metrics_path = 'models/metrics.json'
    
    print("==================================================")
    print("  Starting House Price Prediction ML Pipeline     ")
    print("==================================================")
    
    # 1. Preprocessing & Data Scaling
    print("\n[Step 1] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline(
        dataset_path, test_size=0.2, random_state=42, use_log_target=True
    )
    print(f"Dataset successfully split and scaled.")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Testing shape: {X_test.shape}")
    print(f"  Features engineered ({len(feature_names)}): {feature_names}")
    
    # 2. Model Training and Comparison
    print("\n[Step 2] Training and comparing regressors...")
    best_model, best_model_name, tuned_results = train_and_tune_models(X_train, y_train, cv=5)
    
    # 3. Save Model Artifacts
    print("\n[Step 3] Saving best model and scaler...")
    save_model_artifacts(best_model, scaler, feature_names, model_path, scaler_path, features_path)
    
    # 4. Model Evaluation
    print("\n[Step 4] Evaluating best model on test set...")
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Compute metrics
    metrics = calculate_metrics(y_test, y_pred, is_log_scale=True)
    
    print("\n--- Model Performance Summary (Original Price Scale) ---")
    print(f"Selected Model: {best_model_name}")
    print(f"R2 Score:  {metrics['r2']:.4f}")
    print(f"MAE:       INR {metrics['mae']:,.2f}")
    print(f"RMSE:      INR {metrics['rmse']:,.2f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    
    print("\n--- Model Performance Summary (Log Scale) ---")
    print(f"Log R2:    {metrics['log_r2']:.4f}")
    print(f"Log MAE:   {metrics['log_mae']:.4f}")
    print(f"Log RMSE:  {metrics['log_rmse']:.4f}")
    
    # Save metrics json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics exported to {metrics_path}")
    
    # 5. Generate and Save Visualizations
    print("\n[Step 5] Creating performance visualizations...")
    plot_and_save_results(
        y_test, y_pred, best_model, feature_names, best_model_name, is_log_scale=True, save_dir='plots'
    )
    
    # Output comparison table of CV scores
    print("\n==================================================")
    print("  Model Cross-Validation Score Comparison (R2)  ")
    print("==================================================")
    cv_comparison = []
    for name, data in tuned_results.items():
        cv_comparison.append({
            "Model": name,
            "CV R2 Score": data['score'],
            "Best Parameters": str(data['params'])
        })
    cv_df = pd.DataFrame(cv_comparison)
    print(cv_df.to_string(index=False))
    print("==================================================")
    print("  Pipeline Completed Successfully!                ")
    print("==================================================")

if __name__ == '__main__':
    run_pipeline()
