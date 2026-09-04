# FC24 Player Analytics — ML Capstone Project

Machine Learning capstone project (23CSE301) applying a full end-to-end ML pipeline —
regression, classification, and clustering — to EA Sports FC 24 player data.

## Team

- A Adhithyan- CB.SC.U4CSE24201
- Amal U - CB.SC.U4CSE24204
- Nishitha Kamalapathi -CB.SC.U4CSE24238

## Dataset

**Source:** [EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) (Kaggle)

The raw dataset (`male_players.csv`) contains player records spanning FIFA 15 through FC 24
in a single file, distinguished by a `fifa_version` column. We filter to `fifa_version == 24`
only, giving **18,250 players** with **109 raw attributes** each, including overall rating,
market value, wage, physical attributes, detailed skill sub-attributes (attacking, skill,
movement, power, mentality, defending, goalkeeping), and club/nationality metadata.

After cleaning (dropping identifier/URL/redundant position-rating columns, handling
structural missingness, dropping a leakage-prone column, and engineering one composite
feature), the working dataset (`data/players_fc24_clean.csv`) contains **18,250 rows and
56 columns**.

## Problem Statements

### 1. Regression — Predict Player Market Value
**Target:** `value_eur` (log-transformed to correct right-skew)
**Goal:** Predict a player's market value in euros from their skill attributes, physical
profile, and reputation, using 10 regression algorithms.

### 2. Classification — Predict Player Position Group
**Target:** `position_group` — one of Goalkeeper / Defender / Midfielder / Forward, derived
from each player's primary listed position (`player_positions`).
**Goal:** Classify a player's role from their skill attribute profile alone, using 10
classification algorithms across two review phases.

### 3. Clustering — Discover Playing-Style Archetypes
**No target used during fitting.** Players are grouped into `k=4` clusters based solely on
skill attributes (pace, shooting, passing, dribbling, defending, physical, and their
sub-attributes), using K-Means and Agglomerative Hierarchical Clustering. Ground-truth
position labels are used only afterward, to interpret what each discovered cluster represents.

## Results Summary

### Regression (target: `log1p(value_eur)`)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting Regressor (tuned)** | **0.9978** | — | — |
| Random Forest Regressor (tuned) | 0.9976 | — | — |
| Random Forest Regressor | 0.9975 | 0.0604 | 0.0586 |
| Gradient Boosting Regressor | 0.9959 | 0.0781 | 0.0586 |
| Support Vector Regressor (SVR) | 0.9859 | 0.1444 | 0.1005 |
| Polynomial Regression (deg=2) | 0.9854 | 0.1472 | 0.1067 |
| Decision Tree Regressor | 0.9843 | 0.1524 | 0.1061 |
| Linear Regression | 0.9703 | 0.2098 | 0.1517 |
| Ridge Regression | 0.9703 | 0.2098 | 0.1515 |
| Lasso Regression | 0.9685 | 0.2162 | 0.1506 |
| ElasticNet Regression | 0.9682 | 0.2170 | 0.1513 |
| K-Nearest Neighbors Regressor | 0.9269 | 0.3292 | 0.2448 |

**Best model: Tuned Gradient Boosting Regressor** (`learning_rate=0.05, max_depth=5,
n_estimators=200`), Test R²=0.9978.

### Classification (target: `position_group`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Support Vector Machine (SVC)** | **0.9068** | 0.9087 | 0.9068 | 0.9065 | 0.9702 |
| Logistic Regression | 0.8964 | 0.8970 | 0.8964 | 0.8964 | 0.9777 |
| Random Forest Classifier | 0.8921 | 0.8930 | 0.8921 | 0.8916 | 0.9768 |
| Gradient Boosting Classifier | 0.8915 | 0.8918 | 0.8915 | 0.8913 | 0.9788 |
| Bagging Classifier | 0.8877 | 0.8889 | 0.8877 | 0.8873 | 0.9762 |
| MLP Classifier | 0.8704 | 0.8701 | 0.8704 | 0.8701 | 0.9710 |
| K-Nearest Neighbors | 0.8699 | 0.8725 | 0.8699 | 0.8701 | 0.9600 |
| Decision Tree Classifier | 0.8474 | 0.8479 | 0.8474 | 0.8474 | 0.9293 |
| AdaBoost Classifier | 0.8068 | 0.8141 | 0.8068 | 0.8048 | 0.8611 |
| Naive Bayes (Gaussian) | 0.7921 | 0.7926 | 0.7921 | 0.7912 | 0.9281 |

**Best model: Support Vector Machine (SVC)**, tuned to `C=10, gamma='auto', kernel='rbf'`,
Test Accuracy=0.9068. Gradient Boosting and Logistic Regression have marginally higher
ROC-AUC, indicating better-calibrated probabilities, but SVM wins on the primary hard-label
metrics.

### Clustering (k=4, no labels used during fitting)

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---|---|---|
| **K-Means** | **0.2756** | **1.2501** | **11,577** |
| Agglomerative Hierarchical (ward) | 0.2322 | 1.3706 | 10,211 |

K-Means outperforms Agglomerative Clustering on all three metrics. Both algorithms cleanly
isolate a pure Goalkeeper cluster; K-Means additionally recovers an interpretable
**attacking vs. defensive vs. central** structure among outfield players when cross-tabulated
against true position labels.

## Repository Structure

```
/
├── README.md                Project overview, results, setup (this file)
├── requirements.txt         Python dependencies
├── data/
│   ├── players_fc24_clean.csv   Cleaned, filtered working dataset
│   └── male_players.csv         Raw Kaggle download (not committed — see below)
├── notebooks/
│   ├── regression.ipynb     Full Regression track
│   ├── classification.ipynb Full Classification track (Parts A & B)
│   └── clustering.ipynb     Full Clustering track
├── models/                  Saved model files (optional)
└── app/                     GUI / deployment code (bonus, optional)
```

## Setup & How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nishitha2006/fc24-player-analytics-ml.git
   cd fc24-player-analytics-ml
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\Activate.ps1
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get the dataset**
   Download `male_players.csv` from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
   and place it in `data/`. Alternatively, `data/players_fc24_clean.csv` (already cleaned
   and filtered to FC 24) is committed to the repo and is sufficient to run all three
   notebooks directly without re-downloading the raw file.

5. **Launch Jupyter and run the notebooks**
   ```bash
   jupyter notebook
   ```
   Open `notebooks/regression.ipynb`, `notebooks/classification.ipynb`, and
   `notebooks/clustering.ipynb`, and run each top to bottom (Kernel → Restart & Run All).

## Key Design Decisions

- **Log-transform on `value_eur`**: corrects heavy right-skew in market value (a small
  number of elite players vs. the majority at lower values).
- **`release_clause_eur` dropped**: strongly derived from `value_eur`, would leak the
  regression target.
- **Goalkeeping vs. outfield attributes**: missing values here are structural, not random
  (a player is either a GK or isn't) — imputed with 0 rather than a statistical measure.
- **`position_group` (4-class) target**: built from each player's primary listed position,
  simplified from ~15 raw position codes into Goalkeeper/Defender/Midfielder/Forward for a
  cleaner multi-class problem.
- **Clustering excludes `overall`, `value_eur`, `position_group`**: only raw skill
  attributes are used during fitting, consistent with unsupervised learning; position
  labels are used only afterward for interpretation.

## Academic Integrity / AI Assistance

Generative AI (Claude) was used for code scaffolding, debugging assistance (e.g. resolving
scikit-learn version compatibility issues), and structuring the analysis pipeline. All
dataset-specific analysis, interpretation of results, and feature engineering decisions were
reviewed and written by the team.
