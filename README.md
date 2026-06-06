# Consumer Churn Classification

An Artificial Intelligence laboratory project focused on building an end-to-end machine learning pipeline to predict consumer churn based on demographic and behavioral attributes.

---

## 📌 Project Overview
Consumer churn is a critical challenge across industries because it directly impacts business profitability. This project implements a full machine learning pipeline—encompassing dataset preprocessing, exploratory data analysis (EDA), supervised model training, and unsupervised clustering—to predict whether a consumer will churn or not. 

The primary objective was to establish the pipeline and rigorously evaluate predictive models, identifying how dataset properties and weak feature correlations influence model generalizability.

---

## 👥 Authors (Group 5)
* **Apurbo Sarkar** (ID: 22299379) - *BRAC University*
* **Md. Mushroor Muttakin Khan** (ID: 22201631) - *BRAC University*

---

## 📊 Dataset Profile
* **Dataset Size:** ~1,500 records (rows)
* **Problem Type:** Binary Classification
* **Target Variable (`Churn`):** * `0` = Not churned
  * `1` = Churned
* **Features Matrix:**
  * **Quantitative (Numerical):** `Age`, `Income`, `Credit_Score`, `Num_Purchases`, `Membership_Years`.
  * **Categorical:** `Gender`, `Marital_Status`, `Device_Used`.

---

## 🛠⚙️ Data Preprocessing Pipeline
To resolve data anomalies and prepare features for optimal algorithm performance, the following preprocessing steps were systematically applied:

1. **Handling Invalid & Missing Values:**
   * Numerical constraints were applied to isolate anomalies (e.g., age $< 0$ or $> 120$, credit scores outside the standard $300	ext{--}850$ window, and negative membership years). These outliers were transformed to null markers.
   * **Numerical Imputation:** Handled using a **K-Nearest Neighbors (KNN) Imputer** to deduce missing values based on spatial similarities.
   * **Categorical Imputation:** Replaced missing values using the **Most Frequent** (mode) strategy.
2. **Categorical Encoding:**
   * Features like `Gender`, `Marital_Status`, and `Device_Used` were transformed into a numerical format via **One-Hot Encoding** to generate binary indicators.
3. **Feature Scaling:**
   * Since distance-based models (KNN) and Multi-Layer Perceptrons (MLPs) are highly sensitive to numeric ranges, all numerical features were normalized using **MinMaxScaler** into a fixed boundaries scale of `[0, 1]`.

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights
* **Dataset Balance:** A target variable distribution analysis confirmed that both churn and non-churn classes are virtually equal in size. Consequently, no artificial oversampling (e.g., SMOTE) or undersampling was needed.
* **Feature Distributions:** Distribution plots revealed explicit skewness in `Income` and `Num_Purchases`, while boxplots isolated explicit data outliers in `Income` and `Age`.
* **Predictive Signal Strengths:** A compiled correlation heatmap across numerical parameters indicated remarkably weak correlations between the target variable (`Churn`) and all input features (e.g., `Credit_Score` and `Income` were only marginally related to churn). This high feature overlap between the two target classes explains why standard linear or distance boundaries struggle to partition the data easily.

---

## 📐 Dataset Splitting
The preprocessed data was partitioned via **Stratified Sampling** to guarantee identical target class distributions across all subsets:
* **Training Set:** 70%
* **Validation Set:** 15%
* **Testing Set:** 15%

---

## 🤖 Model Implementations & Configurations

### A. Supervised Learning
1. **K-Nearest Neighbors (KNN):**
   * **Hyperparameter Tuning:** $k = 5$ neighbors.
   * **Observation:** Performance suffered due to high class overlaps in the local multi-dimensional feature space.
2. **Decision Tree Classifier:**
   * **Hyperparameter Tuning:** `max_depth = 5`, `min_samples_split = 10`.
   * **Observation:** Pruning was crucial; larger depths immediately led to overfitting without capturing real predictive patterns.
3. **Neural Network (Multi-Layer Perceptron):**
   * **Architecture Layers:** Hidden layer configuration of `[(32,), (64,), (64, 32)]`.
   * **Activation Function:** `ReLU`
   * **Optimization Algorithm:** `Adam` (Maximum iterations bound: `500`).
   * **Observation:** Performance hovered near random guessing due to uninformative feature mappings.

### B. Unsupervised Learning
* **K-Means Clustering:**
   * **Optimization:** Evaluated via Silhouette Score where $k = 2$ was verified as optimal.
   * **Silhouette Score:** `0.3235` (indicating poor spatial cluster separation).
   * **Pseudo-Accuracy Evaluation:** Mapping the clusters to actual labels by majority vote produced a validation accuracy of ~48% and a test accuracy of ~50%, verifying that natural geometric clusters do not correspond cleanly with churn parameters.

---

## 📉 Evaluation Summary & Benchmarks

The predictive performance on the test set across all evaluated frameworks is summarized below:

| Model Architecture | Test Accuracy | Precision / Recall | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | ~53.00% | ~0.50 / ~0.50 | ~0.55 |
| **Decision Tree** | ~53.44% | ~0.52 / ~0.51 | ~0.54 |
| **Neural Network (MLP)** | ~52.44% | ~0.52 / ~0.53 | ~0.51 |
| **K-Means (Pseudo-Labeling)**| ~50.00% | Consolidated Baseline | — |

### 🛑 Performance Analysis & Challenges
All models scored close to the 50% baseline marker (equivalent to a random guess). The underlying challenge is entirely attributed to **dataset quality and weak feature predictive signaling**, rather than implementation flaws. The feature feature overlaps severely inhibit high-margin geometric boundaries. 

---

## 🔮 Future Enhancements
To scale predictive accuracy beyond current thresholds, future versions should explore:
1. **Advanced Ensemble Methods:** Implementing robust ensemble learning variants like **Random Forest** or **Gradient Boosting (XGBoost/LightGBM)** to find complex non-linear relationships.
2. **Feature Engineering:** Developing derived interaction properties (e.g., custom combinations like `Purchases_per_year` or `Income_per_purchase`) to see if stronger separation patterns emerge.
3. **Data Augmentation:** Collecting richer contextual behavioral information to reduce current distribution overlaps.
