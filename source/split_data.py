import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
csv_path = "all_mfcc_features.csv"  # Adjust if needed
df = pd.read_csv(csv_path)

print("Full dataset shape:", df.shape)

# Split dataset: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Save the splits
train_df.to_csv("train_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)

print("Saved train_features.csv and test_features.csv")
