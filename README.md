# Predicting Battery Performance Using Synthetic Dataset

## Objective
This project aims to develop a predictive model to assess battery performance based on a synthetic dataset that simulates various battery conditions and usage scenarios. The goal is to predict key performance indicators such as battery life and efficiency.

## Techniques Used
1. **Synthetic Data Generation**: A synthetic dataset is generated to represent different battery characteristics and operational environments.
2. **Regression Models**:
   - **Linear Regression**: A baseline model for predicting battery performance.
   - **Random Forest Regressor**: A more robust model to capture non-linear relationships and improve prediction accuracy.
3. **Model Explainability**: SHAP (SHapley Additive exPlanations) is used to analyze feature importance and provide insights into the factors influencing the battery performance predictions.

## Features of the Python Script
The project is implemented in a single Python file where:
- **Training**: Both Linear Regression and Random Forest Regressor models are trained on the synthetic dataset.
- **Prediction**: Predictions are made on a test set, and performance is evaluated using metrics such as R² and Mean Absolute Error.
- **Evaluation**: Evaluation metrics for both models are printed, allowing you to compare their performance.
- **Explainability**: SHAP is used to explain feature importance, helping to understand which features most affect the prediction of battery life and efficiency.

## Dataset
- The dataset is synthetic, representing various battery conditions (e.g., temperature, voltage, cycles, etc.) and performance metrics.

## Instructions to Run the Project
1. Clone the repository.
2. Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python battery_performance.py
   ```
The script will:
  Train the models.
  Output the predictions and performance metrics.
  Display SHAP plots for feature importance.

## Results
Linear Regression provides baseline predictions of battery performance.
Random Forest Regressor offers improved performance by handling non-linear patterns in the dataset.
SHAP Analysis provides insights into which features contribute the most to battery performance predictions.
