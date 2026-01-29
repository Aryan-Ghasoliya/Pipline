# Pipline
# Production-Grade ML Pipeline

This project implements a complete end-to-end production-grade machine learning pipeline for a classification problem using Python. The entire pipeline is implemented in a single Python file while maintaining modularity through functions. It follows industry-standard ML practices and avoids AutoML and black-box libraries.

The pipeline includes data validation, feature engineering, preprocessing, model training with cross-validation, evaluation using proper metrics, model persistence, and full reproducibility using fixed random seeds.

The input dataset must be a CSV file named `dataset.csv` containing the columns `age`, `salary`, `experience`, and `target`. The target column should be binary (0 or 1), and the dataset should contain at least 100 rows for reliable training.

Logistic Regression is used as the model due to its interpretability, stability, efficiency, and suitability as a strong baseline classifier for production systems.

Model performance is evaluated using accuracy, precision, recall, and F1-score. Cross-validation is performed using K-Fold validation to ensure robustness and reduce overfitting.

To run the project, first install the required dependencies using `pip install pandas scikit-learn joblib`. Then execute the pipeline using `python ml_pipeline.py`.

The output includes cross-validation accuracy and evaluation metrics printed to the console, and the trained model is saved to disk as `trained_model.pkl`.

This implementation satisfies all constraints: no AutoML, no end-to-end black-box libraries, single-file execution, reproducibility, and production-ready design.

Author: Aryan Ghasoliya
