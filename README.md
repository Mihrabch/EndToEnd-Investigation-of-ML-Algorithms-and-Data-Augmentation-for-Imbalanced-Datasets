# EndToEnd-ML-Algorithms-and-Data-Augmentation-for-Imbalanced-Datasets

## My Contributions

- Designed and implemented the full end-to-end ML pipeline
- Performed data cleaning and feature engineering on 260k+ records across 20 health and socioeconomic variables
- Implemented and compared four data augmentation strategies tailored to nominal/categorical data
- Conducted hyperparameter tuning on HPCC (Texas Tech University) and model evaluation
- Led manuscript writing — published in Healthcare Analytics, Elsevier 2024

**Published:** Healthcare Analytics, Elsevier, 2024 · [Paper Link](https://doi.org/10.1016/j.health.2023.100297)

---

## Problem

Nearly 1 in 10 Americans has diabetes, but many cases go undiagnosed. Can we use
behavioral and socioeconomic survey data to flag at-risk individuals before clinical
diagnosis? A key challenge: the dataset is heavily imbalanced, with diabetic cases
representing only 18% of records — causing standard ML models to systematically
miss positive cases despite high overall accuracy.

---

## Dataset

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021
- **Raw size:** 438,693 records, 303 attributes
- **After preprocessing:** 262,958 records, 20 variables
- **Class distribution:** 18% diabetic (46,944), 82% non-diabetic (216,014)
- **Challenge:** Severe class imbalance — standard models achieve high accuracy but systematically miss diabetic cases
- **Feature types:** All nominal/categorical — BMI, Age, Income, Blood Pressure, Cholesterol, Heart Disease, Smoking, Kidney Disease, Arthritis, Depression, Exercise, and more

---

## Approach

1. **Data Cleaning** — removed missing values, "Don't Know/Refused/Blank" responses, and selected 20 clinically relevant features — reducing to 262,958 records
3. **Train/Test Split** — 80/20 stratified split, preserving class proportions (18%/82%)
4. **Data Augmentation on Training Data Only** (to prevent data leakage) using four techniques suited for nominal data:
   - SMOTE-N (oversampling)
   - ENN — Edited Nearest Neighbors (undersampling)
   - SMOTE-ENN (hybrid)
   - SMOTE-Tomek (hybrid)
5. **One-Hot Encoding** — applied after augmentation; expanded features from 20 to 92 columns
6. **Model Training** — Logistic Regression, Random Forest, Gradient Boosting, AdaBoost
7. **Ensemble** — Soft Voting Classifier combining top-3 models per sampling strategy
8. **Hyperparameter Tuning** — GridSearchCV (5-fold CV, scoring=recall) run on TTU HPCC
9. **Evaluation** — Precision, Recall, Accuracy, AUC-ROC; primary focus on **recall** given the clinical cost of false negatives

---

## Results

- Baseline models achieved high accuracy (~83%) but poor recall (~57%) — systematically missing diabetic cases
- All augmentation strategies improved recall significantly over the baseline
- **Best result: ENN + Gradient Boosting — Recall 71.7%, AUC 0.791** (~14% recall improvement over baseline)
- Ensemble (Voting Classifier) with ENN further stabilized performance at Recall 71.4%, AUC 0.789
- Key risk factors identified: **Age, BMI, and Blood Pressure** — consistent with clinical literature

> Full results and comparison tables available in the published paper.

---

## Repository Structure

```
├── data/                        # Preprocessed BRFSS dataset
├── brfss_diabetes_pipeline.py     # Clean end-to-end pipeline script (HPCC sample)
└── README.md
```

---

## Citation

Chowdhury M.M., Ayon R.S., Hossain M.S. "An Investigation of Machine Learning
Algorithms and Data Augmentation Techniques for Diabetes Diagnosis Using Class
Imbalanced BRFSS Dataset." *Healthcare Analytics*, Elsevier, 2024.
https://doi.org/10.1016/j.health.2023.100297
