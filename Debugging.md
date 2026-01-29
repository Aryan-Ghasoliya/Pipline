
## Overview
This task demonstrates how to **identify, debug, and fix instability issues** in a machine learning model. The observed problems include **high variance across runs** and **unstable predictions for identical inputs**. The solution shows a comparison between an unstable pipeline (before fixes) and a stabilized pipeline (after fixes) using code-based evidence.

## Problem Symptoms
- Different results on each run
- Inconsistent predictions for the same input
- Unreliable evaluation metrics

## Root Cause Analysis
The instability is caused by:
- Data leakage due to preprocessing before train-test split
- Uncontrolled randomness (no fixed random seeds)
- Lack of regularization
- No cross-validation
- Inconsistent preprocessing across runs

## Debug Checklist
- Ensure train-test split happens before preprocessing  
- Fit preprocessing steps only on training data  
- Fix random seeds everywhere  
- Use cross-validation  
- Apply regularization to reduce overfitting  
- Evaluate only on unseen test data  
- Verify feature consistency  

## Fixes Implemented
1. **Data Leakage Removal**  
   Preprocessing (imputation and scaling) is applied only after splitting the data and fitted exclusively on training data.

2. **Randomness Control and Regularization**  
   Fixed random seeds are used, regularization strength is applied, and cross-validation is added to stabilize performance.

## Before vs After Comparison
**Before Fix**
- Accuracy varies significantly across runs
- No reproducibility
- High variance

**After Fix**
- Consistent accuracy across runs
- Stable predictions
- Cross-validation accuracy aligned with test performance

## Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1-score


