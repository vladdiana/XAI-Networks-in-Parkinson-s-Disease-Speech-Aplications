import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
IN_CSV = os.path.join(BASE, "all_mfcc_features.csv")

df = pd.read_csv(IN_CSV)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_df.to_csv(os.path.join(BASE, "train_features.csv"), index=False)
test_df.to_csv(os.path.join(BASE, "test_features.csv"), index=False)

print("✅ Data split complete for DDK")
print(f"🟢 Train: {train_df.shape[0]}  |  🔵 Test: {test_df.shape[0]}")
