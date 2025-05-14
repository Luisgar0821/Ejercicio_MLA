#  Product Condition Classifier – E-commerce ML Project

##  Project Overview

This project aims to predict whether a product listed on an e-commerce platform is **new** or **used**, based on its structured metadata. Using a real-world dataset in `.jsonl` format with 100,000 product entries, we clean, transform, and model the data to classify the product condition with machine learning.

---

##  Objective

- Clean and transform the raw dataset into a usable format.
- Explore the data to understand feature relevance and patterns.
- Train and evaluate multiple classification models.
- Compare model performance to identify the best-performing approach.

---

##  Dataset Summary

- **Original format:** JSON Lines (`.jsonl`)
- **Total entries:** 100,000 products
- **Cleaned dataset size:** 95,398 entries, 19 features
- **Target variable:** `condition` → 0 = New, 1 = Used

---

##  Preprocessing Pipeline

1. Flattened nested structures: `shipping`, `seller_address`, `pictures`, etc.
2. Engineered new features:
   - `price_diff` = `price - base_price`
   - `num_pictures`, `title_length`, `num_non_mp_methods`
3. Encoded categorical variables with `LabelEncoder`.
4. Removed high-cardinality or text-heavy fields (`title`, `id`).
5. Dropped rows with missing values (`dropna()`).

 Cleaned dataset: `products_clean_final.csv`

---

##  Exploratory Data Analysis (EDA)

- Products were slightly imbalanced: more **new** than **used**.
- `Used` products tend to have higher `sold_quantity` and longer titles.
- Strong correlations found between `price`, `base_price`, and `price_diff`.
- Features like `free_shipping` and location also influenced the condition.

---

##  Models Trained

We trained and compared **three different classification models**:

| Model                  | F1 Score (`New`) | F1 Score (`Used`) | F1 Weighted | Accuracy |
|------------------------|------------------|-------------------|-------------|----------|
| Logistic Regression    | 0.64             | 0.71              | 0.67        | 0.68     |
| Random Forest Classifier | 0.69           | 0.69              | 0.69        | 0.69     |
| Gradient Boosting Classifier | **0.82**   | **0.81**          | **0.82**    | **0.82** |

---

##  Final Conclusion

The **Gradient Boosting Classifier** significantly outperformed the other models in every metric:

- Highest overall accuracy (82%)
- Balanced and high F1-scores for both classes
- Robust performance without overfitting

 **Gradient Boosting is the recommended production model** for this task.

---
##  Project Structure

| File / Folder                         | Description |
|--------------------------------------|-------------|
| `EDA_01.ipynb`                       | Exploratory Data Analysis and feature engineering |
| `model_training.ipynb`              | Unified training pipeline or model experimentation |
| `MLA_100k.jsonlines`                | Original raw dataset in JSONL format |
| `products_clean_final.csv`          | Final cleaned dataset used for model training |
| `products_dataset.csv`              | Intermediate processed version |
| `NewProducts.csv`                   | Possibly new or unseen data (for future predictions) |
| `logreg_product_condition_model.pkl` | Logistic Regression trained model |
| `logreg_scaler.pkl`                 | Scaler for Logistic Regression model |
| `logreg_classification_report.txt`  | Report for Logistic Regression model |
| `product_condition_model.pkl`       | (Possibly older model version, verify its use) |
| `gb_product_condition_model.pkl`    | Gradient Boosting trained model |
| `gb_scaler.pkl`                     | Scaler for Gradient Boosting model |
| `gb_classification_report.txt`      | Report for Gradient Boosting model |
| `classification_report.txt`         | Report for Random Forest (default name) |
| `scaler.pkl`                        | Scaler (likely for Random Forest) |
| `README.md`                         | Project documentation (this file) |
| `.gitignore`                        | Files excluded from version control |
| `venv/`                             | Python virtual environment (not included in Git) |


##  Technologies Used

- Python 3.12
- pandas, numpy
- scikit-learn
- joblib
- seaborn / matplotlib (EDA)

---

##  Author

**Antonio Cárdenas - 2230433**  
Machine Learning Exercise — 2025  
