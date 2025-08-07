import pandas as pd
import os

# Build the path to the CSV file "all_mfcc_features.csv"
# The file is located in the same directory as the current script (using __file__).
csv_path = os.path.join(os.path.dirname(__file__), "all_mfcc_features.csv")

# Read the CSV file into a pandas DataFrame.
# This file contains the extracted MFCC features for all audio files.
df = pd.read_csv(csv_path)

# Display the overall shape of the dataset (number of rows and columns).
print("Dataset shape:", df.shape)

# Display the first 5 rows of the dataset as a quick preview.
print("\nFirst 5 rows:")
print(df.head())

# Count the occurrences of each label (e.g., "HC" vs "PD") to see the dataset balance.
label_counts = df['label'].value_counts()
print("\nLabel distribution (HC vs PD):")
print(label_counts)

# Check for missing values in the entire dataset.
# The result shows the total number of missing entries.
missing = df.isnull().sum().sum()
print("\nMissing values in dataset:", missing)
