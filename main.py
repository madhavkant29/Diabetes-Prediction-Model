import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib

# Loading the data
data = pd.read_csv("data/diabetes.csv")

# Preprocessing
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

# Use separate LabelEncoder for each column
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for later use

# Splitting data
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training RandomForestClassifier with GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Binary ROC-AUC
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
print(f"ROC-AUC: {roc_auc}")

# Cross-validation score
cv_score = cross_val_score(best_model, X_train_scaled, y_train, cv=5).mean()
print(f"Cross-validation score: {cv_score}")

# Saving the model, scaler, and label encoders
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


# Function to preprocess and predict new data
def predict_new_data(input_data: pd.DataFrame):
    # Encode categorical features using saved encoders
    for col in categorical_columns[:-1]:  # Excluding "class"
        if col in label_encoders:
            input_data[col] = input_data[col].apply(
                lambda x: (
                    label_encoders[col].transform([x])[0]
                    if x in label_encoders[col].classes_
                    else -1
                )
            )
    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = best_model.predict(input_scaled)
    return prediction


# New test data
new_data = pd.DataFrame(
    {
        "Gender": ["Male"],
        "Polyuria": ["Yes"],
        "Polydipsia": ["No"],
        "sudden weight loss": ["Yes"],
        "weakness": ["No"],
        "Polyphagia": ["Yes"],
        "Genital thrush": ["No"],
        "visual blurring": ["No"],
        "Itching": ["Yes"],
        "Irritability": ["No"],
        "delayed healing": ["Yes"],
        "partial paresis": ["No"],
        "muscle stiffness": ["Yes"],
        "Alopecia": ["No"],
        "Obesity": ["No"],
        "Age": [35],
    }
)

# Ensure columns match
new_data = new_data[X.columns]

# Predict
prediction = predict_new_data(new_data)
print(f"Prediction for the new data: {prediction[0]}")
