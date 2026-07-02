<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=32&duration=3000&pause=1000&center=true&vCenter=true&width=760&lines=House+Price+Prediction;Machine+Learning+%2B+Regression;Evolving+Into+a+Real-Estate+AI+Platform" />

<br>

[![Top Language](https://img.shields.io/github/languages/top/sreeram0343/House-Price-Prediction?style=for-the-badge&color=blue)](#)
[![Repo Size](https://img.shields.io/github/repo-size/sreeram0343/House-Price-Prediction?style=for-the-badge&color=informational)](#)
[![Last Commit](https://img.shields.io/github/last-commit/sreeram0343/House-Price-Prediction?style=for-the-badge&color=success)](#)
[![License](https://img.shields.io/github/license/sreeram0343/House-Price-Prediction?style=for-the-badge&color=yellow)](#)
[![Stars](https://img.shields.io/github/stars/sreeram0343/House-Price-Prediction?style=for-the-badge&color=orange)](#)
[![Forks](https://img.shields.io/github/forks/sreeram0343/House-Price-Prediction?style=for-the-badge&color=blueviolet)](#)
[![Issues](https://img.shields.io/github/issues/sreeram0343/House-Price-Prediction?style=for-the-badge&color=red)](#)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-ff69b4?style=for-the-badge)](#)

</div>

<br>

<div align="center">

### 🏡 An ML project that predicts residential property prices — and the first milestone of a long-term journey toward a full-scale, AI-powered real estate intelligence platform.

[Explore the Roadmap](#-roadmap) · [Report a Bug](../../issues) · [Request a Feature](../../issues)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Model Performance](#-model-performance)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Connect](#-connect)

---

## 📖 Overview

**House Price Prediction** is a supervised Machine Learning project that estimates residential property prices from structured housing data using **Linear Regression**.

It's intentionally scoped — a clean, well-tested baseline rather than a kitchen-sink notebook — because it's designed to be the **foundation layer** of a much larger build. Every future phase of this project (geospatial modeling, computer vision on listing photos, LLM-powered property insights, and beyond) will be built directly on top of the pipeline established here.

> 📍 **Current phase:** Foundational ML pipeline (data → preprocessing → regression → evaluation → prediction)

---

## ✨ Key Features

- 📂 **CSV Dataset Loading** — ingests structured housing data for processing
- 🧹 **Data Preprocessing** — cleaning and preparation with Pandas
- 🎯 **Feature Selection** — identifying the predictors that matter most
- ✂️ **Train-Test Split** — proper holdout methodology for honest evaluation
- 📈 **Linear Regression Model** — a fast, interpretable baseline built with Scikit-Learn
- ✅ **Model Evaluation** — quantitative performance scoring
- 🔮 **Price Prediction** — generates price estimates on unseen data

---

## 🧰 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,git,github,vscode" />
</p>

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Handling** | Pandas, NumPy |
| **Modeling** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | VS Code / Jupyter Notebook / Streamlit Dashboard |
| **Version Control** | Git & GitHub |

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────┐
│       House Price Dataset (.csv)       │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│     Data Cleaning & Preprocessing      │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│    Feature Engineering & Selection     │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│           Train / Test Split           │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│        Linear Regression Model         │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│   Model Evaluation (R² · MAE · RMSE)   │
└────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│         Predicted House Price          │
└────────────────────────────────────────┘
```

---

## 📁 Project Structure

> *Example layout — rename/reorganize to match your actual repo.*

```text
House-Price-Prediction/
├── data/
│   └── House Price India.csv  # Raw dataset
├── notebooks/
│   └── Notebook.ipynb         # Original EDA & experimentation
├── src/
│   ├── preprocessing.py       # Feature engineering & scaling pipeline
│   ├── train.py               # Model training & tuning (GridSearchCV)
│   └── evaluate.py            # Metrics calculation & plot generation
├── models/
│   ├── best_model.pkl         # Saved best model (XGBoost)
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── features.json          # list of input feature names
│   └── metrics.json           # test set performance metrics
├── plots/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── residuals_distribution.png
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt           # Package dependencies
├── main.py                    # Unified pipeline entry point
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (or conda)

### Installation

```bash
# Clone the repository
git clone https://github.com/sreeram0343/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the default end-to-end pipeline:
```bash
python main.py
```

Configure parameters using CLI options:
```bash
python main.py --cv 3 --test-size 0.15 --random-state 101
```

**CLI Parameter Options:**
* `--dataset`: Custom path to raw dataset CSV file (default: `data/House Price India.csv`).
* `--test-size`: Proportion of test holdout split (default: `0.2`).
* `--random-state`: Seed for partition split & algorithm randomization (default: `42`).
* `--cv`: Number of cross-validation folds during hyperparameter tuning (default: `5`).
* `--no-log-target`: Disable log-transformation on the target price variable.

### 🧪 Running Unit Tests

Run preprocessing feature-engineering unit tests:
```bash
python -m unittest discover -s tests
```

### 📋 Logging & Pipeline Metadata

* **Logs**: All process activities (data loading, schema validations, CV tuning status, models saved) are formatted and stored in `logs/pipeline.log`.
* **Execution Metadata**: Summary records of Python environments, dependency libraries, parameter configs, and final models are serialized to `models/execution_metadata.json` on each pipeline completion.


---

The pipeline automatically trains multiple algorithms (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost) and optimizes hyperparameters. The final selected champion model is **XGBoost**.

### Champion Model (XGBoost) Metrics (Test Set)

| Metric | Score |
|---|---|
| R² Score | `91.01%` (0.9101) |
| Mean Absolute Error (MAE) | `₹64,208.01` |
| Root Mean Squared Error (RMSE) | `₹115,378.61` |
| Mean Absolute Percentage Error (MAPE) | `11.70%` |

### Model Cross-Validation Comparison

| Model | CV R² Score | Hyperparameters Optimized |
|---|---|---|
| **XGBoost** | **0.9043** | `learning_rate: 0.1, max_depth: 7, n_estimators: 100` |
| **Random Forest** | **0.8891** | `max_depth: 15, min_samples_split: 2, n_estimators: 100` |
| **Decision Tree** | **0.8197** | `max_depth: 10, min_samples_split: 10` |
| **Linear Regression** | **0.7878** | *(none)* |
| **Ridge** | **0.7878** | `alpha: 0.1` |
| **Lasso** | **0.7868** | `alpha: 0.001` |

---

## 🗺️ Roadmap

This repo is **Phase 0** of a multi-phase AI Engineering build. Each phase compounds on the last.

### Phase 1 — Deepen the Data Science
- [x] Exploratory Data Analysis (EDA)
- [x] Advanced Feature Engineering
- [x] Advanced Regression Models (Ridge, Lasso, Random Forest, XGBoost)

### Phase 2 — Explainability & Serving
- [x] Feature Importance Insights
- [ ] Explainable AI (SHAP / LIME)
- [ ] FastAPI inference service
- [x] Interactive Dashboard (Streamlit)

### Phase 3 — Spatial & Multimodal Intelligence
- [ ] Geospatial Intelligence (location-aware pricing)
- [ ] Computer Vision (property image analysis)
- [ ] Multimodal Learning (text + image + tabular fusion)

### Phase 4 — Generative & Agentic AI
- [ ] LLM Integration
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] AI Agents for property research & valuation

### Phase 5 — Production & Scale
- [ ] MLOps (CI/CD, model monitoring, versioning, automated retraining)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add some amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- The open-source Python & ML ecosystem — **Pandas**, **NumPy**, and **Scikit-Learn** — for making rapid experimentation possible
- The dataset providers/source used for training and evaluation *(credit here)*

---

## 📬 Connect

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-sreeram0343-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sreeram0343)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](#)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](#)

**⭐ If this project is useful to you, consider starring the repo — it helps a lot!**

</div>
