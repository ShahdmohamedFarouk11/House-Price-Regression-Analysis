# 🏠 House Price Regression Analysis

> A from-scratch implementation of Linear, Ridge, and Lasso Regression to predict house prices , built without relying on sklearn's model classes, covering OLS derivation, gradient descent, and regularization from the ground up.

---

## 📌 Project Overview

This project applies a full regression pipeline on the **King County House Sales** dataset. Three model specifications are built and compared:

- **Single Predictor Regression** — manual OLS derivation (numerical & statistical views)
- **Multiple Linear Regression** — gradient descent with feature selection
- **Ridge Regression** — L2 regularization via gradient descent
- **Lasso Regression** — L1 regularization via gradient descent with lambda tuning

All core models are implemented from scratch in pure NumPy — no sklearn estimators.

---

## 📁 Dataset

**File:** `kc_house_data.csv` — King County, Washington house sales data

| Feature | Type | Description |
|---|---|---|
| `id` | — | Dropped (not predictive) |
| `date` | Date | Sale date → extracted `year` and `month` |
| `price` | Numerical | House sale price (**target**) |
| `bedrooms` | Numerical | Number of bedrooms |
| `bathrooms` | Numerical | Number of bathrooms |
| `sqft_living` | Numerical | Interior living area (sq ft) |
| `sqft_basement` | Numerical | Basement area (sq ft) |
| `waterfront` | Categorical | Waterfront view flag (OHE encoded) |
| `yr_built` | Numerical | Year built → converted to `house_age` |
| `yr_renovated` | Numerical | Year renovated → `was_renovated` flag + `age_since_renovation` |
| `lat`, `long` | Numerical | Geographic coordinates |
| ... | ... | Other structural and location features |

---

## 🗂️ Project Structure

```
regression_project/
│
├── regression_project.ipynb   # Full pipeline notebook
├── kc_house_data.csv          # Raw dataset
└── README.md
```

---

## 🔬 Methodology

### Section 1 — Data Loading & Exploration
- Loaded dataset, inspected types, missing values, and unique value counts
- Dropped `id` (non-predictive); parsed `date` into `year` and `month`

### Section 2 — Feature Engineering
New features created before splitting:

| New Feature | Description |
|---|---|
| `total_sqft` | `sqft_living` + `sqft_basement` |
| `house_age` | `year` − `yr_built` |
| `was_renovated` | Binary flag (1 if ever renovated) |
| `age_since_renovation` | Years since last renovation (0 if never renovated) |

Raw year columns dropped after extraction. `waterfront` cast to categorical for encoding.

### Section 3 — Train/Test Split
80/20 split with `random_state=42` — performed **before** any preprocessing to prevent data leakage.

### Section 4 — Outlier Handling
IQR clipping applied at the 1st–99th percentile range (both target and features):
- Target (`price`) clipped separately on train and test
- All numerical features clipped using train-computed bounds only

### Section 5 — Correlation Analysis
Heatmap computed on training data. Features with `|correlation| > 0.1` with price selected for the multiple regression model.

### Section 6 — Encoding & Scaling
- `waterfront` → One-Hot Encoding (fit on train, transform both)
- All numerical features → `StandardScaler` (fit on train, transform both)

### Section 7 — Exploratory Visualizations
- Scatter plot: `sqft_living` vs `price` with OLS regression line
- Price distribution histogram
- Average price by bedroom count (bar chart)
- KDE density plot of price

---

## 🤖 Modeling

### View 1 — Numerical (Manual OLS)
Single predictor (`sqft_living`) regression computed entirely by hand:

```
B1 = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
B0 = ȳ − B1 × x̄
```

### View 2 — Statistical (From Scratch)
Custom `StatisticalLinearRegression` class implementing the Normal Equation with full inference output: SE(B0), SE(B1), t-statistic, MSE, R².

### View 3 — Machine (Gradient Descent)
Custom `multipleLinearRegressionGd` class — multiple linear regression trained with batch gradient descent on correlation-selected features.

---

## 📐 Ridge Regression (From Scratch)

Custom `RidgeRegressionGD` class — L2 regularization added to gradient update:

```
dw = (1/m) × Xᵀ × error + (2λ/m) × w
```

Lambda tuned over `[1, 0.1, 0.01, 0.001]` — best selected by highest test R².

Convergence plotted interactively with Plotly (loss vs. iterations).

---

## 🔪 Lasso Regression (From Scratch)

Custom `LassoRegressionGD` class — L1 regularization using subgradient (sign of weights):

```
dw = (1/m) × Xᵀ × error + λ × sign(w)
```

Lambda tuned over `[0.0001, 0.001, 0.01, 0.1, 1, 5, 10]` using a 20% validation split from training data — best selected by lowest validation MSE.

**Feature retention:** Lasso kept all 14 selected features (no zeroing), shrinking coefficients of weaker predictors rather than eliminating them.

---

## 📊 Model Comparison

| Model | Predictors | MSE | R² |
|---|---|---|---|
| Multiple Linear Regression | 14 (selected) | — | — |
| Ridge Regression | All numerical | — | — |
| Lasso Regression | 14 (selected) | — | — |

> Exact metric values are printed at runtime since they depend on gradient descent convergence.

### Interactive Visualizations (Plotly)
- **Residual plots** — dropdown to compare all 3 model specifications side by side
- **Coefficient bar charts** — toggle between Single / Ridge / Lasso coefficients
- **Convergence curves** — loss vs. iterations for Ridge and Lasso
- **Lasso lambda tuning curve** — validation MSE vs. lambda (log scale)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | All from-scratch model math |
| `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Static visualizations |
| `plotly` | Interactive convergence & residual plots |
| `sklearn` | Preprocessing only (OHE, StandardScaler, train_test_split) |

---

## 👩‍💼 Author

**Shahd Ahmed Farghaly**
*Data Science Student — Alexandria University*

📧 [shahdfarghaly2005@gmail.com](mailto:shahdfarghaly2005@gmail.com)
🔗 [LinkedIn Profile](https://www.linkedin.com/in/shahd-farghaly-bb9356332)
