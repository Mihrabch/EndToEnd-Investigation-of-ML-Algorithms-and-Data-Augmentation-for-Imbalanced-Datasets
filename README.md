# EndToEnd-ML-Algorithms-and-Data-Augmentation-for-Imbalanced-Datasets

> I led the ML pipeline design, data preprocessing, model development, 
> and analysis for this project.
## My Contributions
- Designed and implemented the full ML pipeline
- Performed data cleaning and feature engineering on 440k+ records
- Implemented and compared SMOTE augmentation variants
- Conducted hyperparameter tuning and model evaluation
- Led manuscript writing — published in Healthcare Analytics, Elsevier 2024

**Published:** Healthcare Analytics, Elsevier, 2024 · [Paper Link](YOUR_DOI_HERE)

## Problem
Nearly 1 in 10 Americans has diabetes, but many cases go undiagnosed. 
Can we use behavioral and socioeconomic survey data to flag at-risk individuals 
before clinical diagnosis?

## Dataset
- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Size:** 440,000+ survey records
- **Challenge:** Severe class imbalance (~14% positive diabetes cases)

## Approach
1. Data cleaning and feature engineering on 20+ health/socioeconomic variables
2. Class imbalance correction using SMOTE variants (SMOTE, BorderlineSMOTE, ENN)
3. Model comparison: Logistic Regression, Random Forest, XGBoost, SVM
4. Hyperparameter tuning via Grid Search
5. Evaluation: ROC-AUC, F1, Precision-Recall

## Results

| Model | ROC-AUC | F1 (Minority) |
|-------|---------|----------------|
| Logistic Regression | 0.82 | 0.61 |
| Random Forest | 0.86 | 0.67 |
| XGBoost + SMOTE | **0.87** | **0.71** |

- Best model: XGBoost with BorderlineSMOTE augmentation
- Key risk factors: BMI, age, physical activity, high blood pressure, income level

## Repository Structure
```
├── data/               # Preprocessed BRFSS dataset
├── notebooks   # Main analysis notebook
└── README.md
```


## Citation
Chowdhury M.M., Ayon R.S., Hossain M.S. "An Investigation of Machine Learning 
Algorithms and Data Augmentation Techniques for Diabetes Diagnosis Using Class 
Imbalanced BRFSS Dataset." Healthcare Analytics, Elsevier, 2024.
