import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib

# Load the data
data = pd.read_csv("data/diabetes.csv")

# Preprocessing
label_encoder = LabelEncoder()
categorical_columns = [
    "Gender",
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
    "class",
]

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Splitting data into features and target
X = data.drop("class", axis=1)
y = data["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PolynomialFeatures (if needed during training)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Model training using RandomForestClassifier with GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train_poly, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_poly)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_poly)[:, 1])
print(f"ROC-AUC: {roc_auc}")

# Cross-validation score
cv_score = cross_val_score(best_model, X_train_poly, y_train, cv=5).mean()
print(f"Cross-validation score: {cv_score}")

# Saving the model, scaler, and polynomial features
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(poly, "poly.pkl")


# Function to predict new data using DataFrame
def predict_new_data(input_data: pd.DataFrame):
    # Ensure the input is in the correct format
    input_scaled = scaler.transform(input_data)

    # Apply polynomial features
    input_poly = poly.transform(input_scaled)

    # Predict using the trained model
    prediction = best_model.predict(input_poly)

    return prediction


# Example usage of prediction with DataFrame
# Ensure new_data columns match the order of the training data columns
new_data = pd.DataFrame(
    {
        "Gender": [0],  # Example values for each feature
        "Polyuria": [1],
        "Polydipsia": [0],
        "sudden weight loss": [1],
        "weakness": [0],
        "Polyphagia": [1],
        "Genital thrush": [0],
        "visual blurring": [0],
        "Itching": [1],
        "Irritability": [0],
        "delayed healing": [1],
        "partial paresis": [0],
        "muscle stiffness": [1],
        "Alopecia": [0],
        "Obesity": [1],
        "Age": [45],  # Ensure 'Age' is included in the new data
    }
)

# Align the new data columns with the training data
new_data = new_data[X.columns]

prediction = predict_new_data(new_data)
print(f"Prediction for the new data: {prediction[0]}")
