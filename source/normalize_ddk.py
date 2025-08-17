import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
train_path = os.path.join(BASE, "train_features.csv")
test_path  = os.path.join(BASE, "test_features.csv")
train_out  = os.path.join(BASE, "train_features_normalized.csv")
test_out   = os.path.join(BASE, "test_features_normalized.csv")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

mfcc_cols = [c for c in train_df.columns if c.startswith("mfcc_")]
if not mfcc_cols:
    raise RuntimeError("No MFCC columns found in DDK CSVs.")

X_train = train_df[mfcc_cols]
y_train = train_df["label"]
X_test  = test_df[mfcc_cols]
y_test  = test_df["label"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

pd.DataFrame(X_train_s, columns=mfcc_cols).assign(label=y_train).to_csv(train_out, index=False)
pd.DataFrame(X_test_s, columns=mfcc_cols).assign(label=y_test).to_csv(test_out, index=False)

print("✅ Normalization complete for DDK")
print(f"→ {train_out}\n→ {test_out}")
