# 📱 Consumer Churn Classification

An AI laboratory project focused on building an end-to-end machine learning pipeline to predict customer churn based on demographic and behavioral data.

---

## 📌 Project Overview
Customer churn happens when customers stop doing business with a company. This project builds a complete machine learning pipeline (including data cleaning, data exploration, supervised model training, and unsupervised clustering) to predict whether a consumer will leave or stay. 

The main goal was to test how well different models perform when the dataset features have very weak connections to the final target.

---

## 👥 Authors (Group 5)
* **Apurbo Sarkar** (ID: 22299379) - *BRAC University*
* **Md. Mushroor Muttakin Khan** (ID: 22201631) - *BRAC University*

---

## 📊 Dataset Profile
* **Dataset Size:** ~1,500 rows
* **Problem Type:** Binary Classification
* **Target Variable (`Churn`):** `0` = Stayed, `1` = Left
* **Features:**
  * **Numerical:** `Age`, `Income`, `Credit_Score`, `Num_Purchases`, `Membership_Years`
  * **Categorical:** `Gender`, `Marital_Status`, `Device_Used`

---

## 🛠️ Data Preprocessing Pipeline
The following steps prepare the raw features for training:

1. **Handling Invalid & Missing Values:**
   * Filtered incorrect data (Age outside 0 to 120, Credit Score outside 300 to 850, negative membership years) and marked them as null.
   * **Numerical Imputation:** Filled missing numbers using a **KNN Imputer** based on data similarities.
   * **Categorical Imputation:** Filled missing text using the **Most Frequent** value (mode).
2. **Categorical Encoding:** Converted text data (`Gender`, `Marital_Status`, `Device_Used`) into 0 and 1 columns using **One-Hot Encoding**.
3. **Feature Scaling:** Normalized all numerical values between `0` and `1` using **MinMaxScaler** to help distance-based models perform correctly.

---

## 🔍 Key Data Exploration Insights
* **Dataset Balance:** The dataset is perfectly balanced between both classes (churned vs. stayed). No oversampling (SMOTE) was needed.
* **Data Distributions:** `Income` and `Num_Purchases` are highly skewed, with clear outliers visible in `Income` and `Age`.
* **Weak Signal Strengths:** A correlation heatmap revealed that input features share almost no relationship with the `Churn` label. Because data points for both classes overlap heavily, standard machine learning boundaries struggle to split the data effectively.

---

## 📐 Dataset Splitting
The data was split using stratified sampling to maintain identical class ratios across all sets:
* **Training Set:** 70%
* **Validation Set:** 15%
* **Testing Set:** 15%

---

## 🤖 Model Implementations & Configurations

### A. Supervised Learning
1. **K-Nearest Neighbors (KNN):** Configured with k = 5. Performance was low due to high class overlaps in the local data space.
2. **Decision Tree:** Limited to `max_depth = 5` and `min_samples_split = 10`. Keeping the tree small was necessary to prevent immediate overfitting.
3. **Neural Network (MLP):** Built with three hidden layers `[32, 64, (64, 32)]`, `ReLU` activation, and the `Adam` optimizer (500 iterations). Results hovered near random guessing due to uninformative features.

### B. Unsupervised Learning
* **K-Means Clustering:** Optimized at k = 2 clusters. It achieved a low Silhouette Score of `0.3235` (heavy cluster overlap) and yielded a poor ~50% test accuracy when mapped to actual labels.

---

## 📉 Evaluation Summary & Benchmarks

| Model Architecture | Test Accuracy | Precision / Recall | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | ~53.00% | ~0.50 / ~0.50 | ~0.55 |
| **Decision Tree** | ~53.44% | ~0.52 / ~0.51 | ~0.54 |
| **Neural Network (MLP)** | ~52.44% | ~0.52 / ~0.53 | ~0.51 |
| **K-Means (Clustering)**| ~50.00% | Baseline Comparison | Baseline |

### ⚠️ Performance Analysis & Challenges
Every model scored close to the 50% baseline mark, which is equivalent to a random coin toss. This performance barrier is caused entirely by **weak feature signals and poor dataset quality** rather than pipeline execution flaws. 

---

## 🔮 Future Enhancements
To break past the 50% accuracy limit, future iterations will focus on:
1. **Advanced Ensemble Methods:** Deploying robust models like **Random Forest**, **XGBoost**, or **LightGBM** to discover complex, non-linear patterns.
2. **Feature Engineering:** Creating derived metrics (e.g., `Purchases_per_year` or `Income_per_purchase`) to find stronger separation lines.
3. **Data Augmentation:** Gathering deeper behavioral data to reduce the current class overlaps.
