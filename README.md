# Diabetes-Prediction-Model
This project aims to predict whether a patient is at risk of diabetes using machine learning techniques. It uses a Random Forest Classifier model trained on a dataset with various health-related features. The model is evaluated using several metrics such as accuracy, classification report, confusion matrix, ROC-AUC score, and cross-validation.

**Key Features**
Data Preprocessing: The dataset is preprocessed by encoding categorical features and scaling numerical features to standardize the data.
Model Training: The model is trained using a Random Forest Classifier, and hyperparameter tuning is performed using GridSearchCV.
Model Evaluation: The model's performance is evaluated using multiple metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation score.
Saving the Model: The trained model, along with the scaler and polynomial transformer, is saved using the joblib library for future use.
Prediction Functionality: The model can be used to predict whether new data corresponds to a diabetic or non-diabetic patient.

**Evaluation Metrics**
The model achieved the following evaluation results on the test data:

Accuracy: 99.36%

Classification Report:
Precision: 0.99
Recall: 0.99
F1-Score: 0.99

Confusion Matrix:
True Positives: 101
False Positives: 0
False Negatives: 1
True Negatives: 54

ROC-AUC: 1.0

Cross-validation Score: 96.7%

**Usage**
To use the trained model for making predictions:

Ensure that you have the necessary dependencies installed (see the requirements.txt below).
Load the trained model, scaler, and polynomial transformer from their saved files.
Prepare the new input data by encoding and scaling it in the same way as the training data.
Use the model to make predictions on the new data.

**Running the Code**
To train the model, simply run the Python script, which will:

Load the data, preprocess it, and train the model.
Evaluate the model on test data and print the evaluation metrics.
Save the trained model and related objects (scaler, polynomial features) for future use.
To make predictions, you can load the saved model and apply it to new data.

