# Sleep Disorder Prediction App

This is a Streamlit web app that allows you to train multiple machine learning models to predict sleep disorders from your dataset, compare their performance, and make real-time predictions using the best model.

## Features

- Upload your sleep health dataset (CSV)
- Data preprocessing with SMOTE balancing
- Train models: SVM, Random Forest, XGBoost, ANN, CNN-LSTM, and a Stacking Ensemble
- Hyperparameter tuning with GridSearchCV
- Accuracy comparison table and bar chart
- Confusion matrix visualization
- SHAP feature importance for model explainability
- Manual input form to predict sleep disorder for new data points

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
