import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load train and test CSV files
train_path = os.path.join(os.path.dirname(__file__), "train_features.csv")
test_path = os.path.join(os.path.dirname(__file__), "test_features.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Original train shape:", train_df.shape)
print("Original test shape:", test_df.shape)

# Select only the MFCC columns (mfcc_1 ... mfcc_13)
mfcc_columns = [col for col in train_df.columns if col.startswith("mfcc_")]

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training set and transform both train and test
train_df[mfcc_columns] = scaler.fit_transform(train_df[mfcc_columns])
test_df[mfcc_columns] = scaler.transform(test_df[mfcc_columns])

# Save normalized datasets
train_df.to_csv("train_features_normalized.csv", index=False)
test_df.to_csv("test_features_normalized.csv", index=False)

print("Normalized features saved to train_features_normalized.csv and test_features_normalized.csv")
