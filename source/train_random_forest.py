import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # pentru a salva modelul

# Load train and test datasets
train_df = pd.read_csv("train_features_normalized.csv")
test_df = pd.read_csv("test_features_normalized.csv")

# Separate features and labels
X_train = train_df.drop(columns=["file_path", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["file_path", "label"])
y_test = test_df["label"]

# Create and train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
print("\nModel saved as random_forest_model.pkl")
