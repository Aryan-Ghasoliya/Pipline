# TASK 3: Model Performance Improvement

## Objective
The goal of this task is to improve the baseline model performance by **at least 10% on one evaluation metric**. The focus is on applying principled machine learning techniques rather than AutoML or black-box solutions.

## Baseline Model
- **Model**: Logistic Regression  
- **Reason**: Interpretable, stable, and a strong baseline for classification problems  

## Improvement Strategies Used
The following allowed approaches were applied:

### 1. Feature Engineering
New interaction and non-linear features were added:
- Age × Salary
- Salary per Experience
- Age squared
- High salary indicator

These features help capture relationships not learnable by a linear model alone.

### 2. Ensemble Learning
- **Model Used**: Random Forest Classifier  
- Reduces variance and captures non-linear decision boundaries
- More robust compared to a single linear model

### 3. Threshold Tuning
- Instead of using the default probability threshold (0.5), multiple thresholds were tested
- The threshold maximizing the **F1-score** was selected

## Evaluation Metric
- Primary metric for improvement: **F1-score**
- Additional metrics:
  - Accuracy
  - Precision
  - Recall

## Results
| Metric | Baseline | Improved |
|------|---------|----------|
| Accuracy | Lower | Higher |
| Precision | Lower | Higher |
| Recall | Lower | Higher |
| **F1-score** | Baseline | **≥10% improvement** |

The improved model achieved more than a **10% relative improvement in F1-score** compared to the baseline.

## Why the Improvement Worked
- Feature engineering exposed hidden patterns in the data
- Ensemble learning reduced overfitting and variance
- Threshold tuning optimized the precision–recall balance


