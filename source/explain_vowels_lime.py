import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ===== Paths =====
BASE_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\vowels"
TRAIN_CSV = os.path.join(BASE_DIR, "train_features_normalized.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_features_normalized.csv")
OUT_DIR   = os.path.join(BASE_DIR, "explainability", "lime")
os.makedirs(OUT_DIR, exist_ok=True)

# NOTE: LIME works best with a scikit-learn model.
# If you want LIME on NN, you can wrap a predict_proba() around your Torch model.
# Here I'll use a quick Logistic Regression baseline trained on MFCCs for the demo.
from sklearn.linear_model import LogisticRegression

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

feature_cols = [c for c in train_df.columns if c.startswith("mfcc_")]
X_train = train_df[feature_cols].values
y_train = train_df["label"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["label"].values

# Train a simple LR for LIME explanation (fast + has predict_proba)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_cols,
    class_names=["HC", "PD"],
    mode="classification"
)

idx = 0
exp = explainer.explain_instance(
    data_row=X_test[idx],
    predict_fn=clf.predict_proba,
    num_features=min(10, len(feature_cols))
)

print(f"Explanation for test sample {idx}:")
print(exp.as_list())

fig = exp.as_pyplot_figure()
plt.title(f"LIME (VOWELS) - sample {idx}")
out_png = os.path.join(OUT_DIR, "lime_sample0.png")
plt.savefig(out_png, bbox_inches='tight'); plt.close()

print(f"✅ Saved LIME explanation to: {out_png}")
