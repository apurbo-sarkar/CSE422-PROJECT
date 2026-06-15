# Consumer Churn Classification

An Artificial Intelligence laboratory project focused on building an end-to-end machine learning pipeline to predict consumer churn based on demographic and behavioral attributes.

---

## 📌 Project Overview
Consumer churn happens when customers stop doing business with a company, which directly hurts profitability. This project builds a complete machine learning pipeline to predict whether a customer will leave or stay. The pipeline includes data cleaning, exploratory data analysis (EDA), supervised learning models, and unsupervised clustering. 

The main goal was to build the pipeline and test how well different models perform when dealing with weak data correlations and overlapping customer traits.

---

## 👥 Authors (Group 5)
* **Apurbo Sarkar** (ID: 22299379) - *BRAC University*
* **Md. Mushroor Muttakin Khan** (ID: 22201631) - *BRAC University*

---

## 📊 Dataset Profile
* **Dataset Size:** Around 1,500 rows
* **Problem Type:** Binary Classification
* **Target Variable (`Churn`):** * `0` = Stayed (Not churned)
  * `1` = Left (Churned)
* **Features Matrix:**
  * **Numerical Features:** `Age`, `Income`, `Credit_Score`, `Num_Purchases`, `Membership_Years`
  * **Categorical Features:** `Gender`, `Marital_Status`, `Device_Used`

---

## 🛠⚙️ Data Preprocessing Pipeline
To fix errors in the data and prepare it for the machine learning models, we followed these steps:

1. **Handling Invalid and Missing Values:**
   * We set strict rules to catch unrealistic numbers, such as ages below 0 or above 120, credit scores outside the 300 to 850 range, and negative membership years. These errors were turned into null markers.
   * **Missing Numbers:** Filled using a **K-Nearest Neighbors (KNN) Imputer**, which estimates missing values based on similar rows.
   * **Missing Categories:** Filled using the **Most Frequent** value (mode) strategy.
2. **Categorical Encoding:**
   * We converted text categories like `Gender`, `Marital_Status`, and `Device_Used` into numbers using **One-Hot Encoding** to create simple binary columns.
3. **Feature Scaling:**
   * Distance-based models like KNN and Neural Networks are sensitive to data scales. We normalized all numerical features using **MinMaxScaler** to fit them within a clean `[0, 1]` range.

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights
* **Dataset Balance:** The target variable shows that churned and non-churned classes are almost equal in size. Because the data is already balanced, we did not need to use resampling techniques like SMOTE.
* **Data Distributions:** Distribution plots showed that `Income` and `Num_Purchases` are skewed, while boxplots highlighted clear outliers in `Income` and `Age`.
* **Weak Signal Strength:** The correlation heatmap revealed very weak relationships between the target variable (`Churn`) and the input features. For example, `Credit_Score` and `Income` barely relate to whether a customer leaves. This high overlap between classes explains why the models struggled to separate the data.

---

## 📐 Dataset Splitting
We split the data using **Stratified Sampling** to ensure that the training, validation, and testing sets all have the exact same ratio of churned to non-churned customers:
* **Training Set:** 70%
* **Validation Set:** 15%
* **Testing Set:** 15%

---

## 🤖 Model Implementations & Configurations

### A. Supervised Learning
1. **K-Nearest Neighbors (KNN):**
   * **Setup:** Tried with $k = 5$ neighbors.
   * **Result:** Performance was low because the customer data points overlap heavily in the feature space.
2. **Decision Tree Classifier:**
   * **Setup:** Limited to a maximum depth of 5 and minimum samples split of 10.
   * **Result:** Pruning the tree was necessary. Deeper trees immediately overfitted the training data without finding real patterns.
3. **Neural Network (Multi-Layer Perceptron):**
   * **Setup:** Used hidden layers structured as `[(32,), (64,), (64, 32)]`, `ReLU` activation, and the `Adam` optimizer for up to 500 iterations.
   * **Result:** Performance remained close to random guessing because the input features lacked clear patterns.

### B. Unsupervised Learning
* **K-Means Clustering:**
   * **Setup:** Evaluated using the Silhouette Score, which confirmed 2 clusters as the best choice.
   * **Result:** The Silhouette Score was `0.3235`, which indicates poor separation between clusters. Mapping these clusters to actual churn labels resulted in an accuracy of about 48% on validation and 50% on testing. This proves that natural geometric groupings do not match up with customer churn behavior.

---

## 📉 Evaluation Summary & Benchmarks

The final performance scores on the test set are summarized below:

| Model Architecture | Test Accuracy | Precision / Recall | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | ~53.00% | ~0.50 / ~0.50 | ~0.55 |
| **Decision Tree** | ~53.44% | ~0.52 / ~0.51 | ~0.54 |
| **Neural Network (MLP)** | ~52.44% | ~0.52 / ~0.53 | ~0.51 |
| **K-Means (Pseudo-Labeling)**| ~50.00% | Baseline Comparison | Baseline |

### 🛑 Performance Analysis & Challenges
All models scored very close to 50%, which is the same as guessing with a coin flip. This difficulty stems entirely from **poor dataset quality and weak predictive signals** rather than coding errors. The heavy overlap between the attributes of customers who stay and customers who leave makes it difficult for any algorithm to draw a clear boundary line.

---

## 🔮 Future Enhancements
To improve prediction accuracy in future versions, we recommend the following steps:
1. **Advanced Ensemble Methods:** Use stronger ensemble algorithms like **Random Forest** or **Gradient Boosting (XGBoost and LightGBM)** to find complex, non-linear patterns.
2. **Feature Engineering:** Create new combined features, such as `Purchases_per_year` or `Income_per_purchase`, to see if they offer better separation between classes.
3. **Data Augmentation:** Collect more detailed behavioral data to reduce the current overlap between the customer groups.
