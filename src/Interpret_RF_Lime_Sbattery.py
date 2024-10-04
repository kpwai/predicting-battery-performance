import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import torch
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

# Load dataset
df = pd.read_csv('Sbattery_performance_dataset.csv')

# Separate features and labels
X = df.drop('Failure', axis=1)
y = df['Failure']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data for future use if needed
torch.save((X_train, X_test, y_train, y_test), 'preprocessed_Sbattery.pt')

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, 'rf_model.joblib')

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# LIME explanation for local interpretability
# Create a LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=df.drop('Failure', axis=1).columns, class_names=['Not Failure', 'Failure'])

# Choose an instance from the test set to explain (e.g., the first instance)
instance_idx = 0  # Example: explaining the first instance

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=df.drop('Failure', axis=1).columns,
    class_names=['No Failure', 'Failure'],  # Replace with your actual class names
    discretize_continuous=True
)

# Explain a prediction
i = 2  # Index of the test instance to explain
exp = explainer.explain_instance(X_test[i], rf_model.predict_proba, num_features=5)  # Use X_test[i] directly

# Save the explanation as an HTML file
explanation_file = 'lime_explanation_Sbattery_performance_RF.html'
exp.save_to_file(explanation_file)