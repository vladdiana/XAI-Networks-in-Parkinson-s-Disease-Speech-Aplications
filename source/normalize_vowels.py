import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Base path
base_dir = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\vowels"

train_path = os.path.join(base_dir, "train_features.csv")
test_path = os.path.join(base_dir, "test_features.csv")
train_out = os.path.join(base_dir, "train_features_normalized.csv")
test_out = os.path.join(base_dir, "test_features_normalized.csv")

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --- Keep ONLY MFCC columns + label ---
mfcc_cols = [c for c in train_df.columns if c.startswith("mfcc_")]
if not mfcc_cols:
    raise RuntimeError("No MFCC columns found (expected columns starting with 'mfcc_').")

X_train = train_df[mfcc_cols]
y_train = train_df["label"]
X_test = test_df[mfcc_cols]
y_test = test_df["label"]

# Normalize MFCC features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save normalized datasets (MFCCs + label)
pd.DataFrame(X_train_scaled, columns=mfcc_cols).assign(label=y_train).to_csv(train_out, index=False)
pd.DataFrame(X_test_scaled, columns=mfcc_cols).assign(label=y_test).to_csv(test_out, index=False)

print("✅ Normalization complete for VOWELS subset")
print(f"💾 Saved to:\n → {train_out}\n → {test_out}")
