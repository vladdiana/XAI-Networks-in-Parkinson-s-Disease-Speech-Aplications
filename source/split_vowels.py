import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths for vowels
base_dir = os.path.join("experiments", "vowels")
input_csv = os.path.join(base_dir, "all_mfcc_features.csv")

# Load dataset
df = pd.read_csv(input_csv)

# Split into train/test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Save outputs
train_df.to_csv(os.path.join(base_dir, "train_features.csv"), index=False)
test_df.to_csv(os.path.join(base_dir, "test_features.csv"), index=False)

print("✅ Data split complete for VOWELS subset")
print(f"🟢 Train: {train_df.shape[0]} samples | 🔵 Test: {test_df.shape[0]} samples")
