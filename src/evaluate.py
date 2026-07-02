import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from src.logger import logger

def calculate_metrics(y_true, y_pred, is_log_scale=True):
    """
    Computes standard regression metrics.
    If targets are log-scaled, computes metrics on both log-scale and original scale.
    """
    metrics = {}
    
    if is_log_scale:
        # Metrics on log scale
        metrics['log_r2'] = r2_score(y_true, y_pred)
        metrics['log_mae'] = mean_absolute_error(y_true, y_pred)
        metrics['log_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Transform back to original scale
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Metrics on original scale
    metrics['r2'] = r2_score(y_true_orig, y_pred_orig)
    metrics['mae'] = mean_absolute_error(y_true_orig, y_pred_orig)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    metrics['mape'] = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
    
    return metrics

def plot_and_save_results(y_true, y_pred, model, feature_names, model_name, is_log_scale=True, save_dir='plots'):
    """
    Generates and saves model evaluation plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Back-transform to original prices for visual interpretation
    if is_log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Plot 1: Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true_orig, y=y_pred_orig, alpha=0.5, color='#4A90E2')
    # Draw reference line
    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='#FF5A5F', lw=2)
    plt.title(f'Actual vs Predicted House Prices ({model_name})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Price (INR)', fontsize=12)
    plt.ylabel('Predicted Price (INR)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # Plot 2: Residuals Distribution
    residuals = y_true_orig - y_pred_orig
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#2ECC71', bins=40)
    plt.axvline(0, color='#FF5A5F', linestyle='--', lw=2)
    plt.title(f'Distribution of Residuals ({model_name})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Residuals (Actual - Predicted Price)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residuals_distribution.png'), dpi=150)
    plt.close()

    # Plot 3: Feature Importances or Coefficients
    plt.figure(figsize=(10, 8))
    
    has_importances = False
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Feature Importances ({model_name})'
        has_importances = True
    elif hasattr(model, 'coef_'):
        importances = model.coef_
        title = f'Model Coefficients ({model_name})'
        has_importances = True
        
    if has_importances:
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 15 features or all if less
        sns.barplot(
            data=feat_imp.head(15),
            x='Importance',
            y='Feature',
            hue='Feature',
            palette='viridis',
            legend=False
        )
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Importance Value / Coefficient Weight', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150)
        plt.close()
    else:
        # Create an empty dummy plot noting that importances are not available
        plt.text(0.5, 0.5, 'Feature Importances not supported for this model type.', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.title('Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150)
        plt.close()

    logger.info(f"Evaluation plots saved to the folder '{save_dir}/'")
