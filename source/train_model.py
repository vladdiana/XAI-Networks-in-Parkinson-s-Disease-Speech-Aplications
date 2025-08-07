import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load normalized features
train_path = "train_features_normalized.csv"
test_path = "test_features_normalized.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["file_path", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["file_path", "label"])
y_test = test_df["label"]

# Create a folder to save reports and plots
os.makedirs("source/reports", exist_ok=True)

def save_results(model_name, y_true, y_pred):
    """Save classification report and confusion matrix for a given model"""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Save report to .txt
    report_path = f"source/reports/{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HC", "PD"], yticklabels=["HC", "PD"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"source/reports/{model_name}_confusion_matrix.png")
    plt.close()

    print(f"Saved report and confusion matrix for {model_name}.")

# ---- Train and evaluate models ----
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

for model_name, model in models.items():
    print(f"\n===== Training {model_name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save results
    save_results(model_name, y_test, y_pred)

print("\nAll models have been trained and reports saved in 'source/reports/' folder.")
