import os
import librosa
import numpy as np
import pandas as pd
import argparse

# =========================
# Parse command-line arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--subset", required=True, help="Subset: vowels, ddk, sentences, monologue")
args = parser.parse_args()
subset = args.subset.lower()

# =========================
# Define paths to each subset
# =========================
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PC-GITA_16kHz"))

subset_paths = {
    "vowels": os.path.join(base_path, "Vowels"),
    "ddk": os.path.join(base_path, "DDK analysis"),
    "sentences": os.path.join(base_path, "sentences2"),
    "monologue": os.path.join(base_path, "monologue")
}

# Check that the provided subset is valid
if subset not in subset_paths:
    raise ValueError(f"Subset '{subset}' not recognized. Use: {list(subset_paths.keys())}")

DATA_DIR = subset_paths[subset]
print(f"📁 Processing audio from: {DATA_DIR}")

# =========================
# Utility: List all .wav files recursively
# =========================
def list_wav_files(root):
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".wav"):
                all_files.append(os.path.join(dirpath, f))
    return all_files

# =========================
# Extract MFCC features from one audio file
# =========================
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length=160, win_length=400)
    return np.mean(mfcc, axis=1)  # Mean over time

# =========================
# Process and save results to CSV
# =========================
output_dir = os.path.join("experiments", subset)
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "all_mfcc_features.csv")

data = []
for file in list_wav_files(DATA_DIR):
    try:
        features = extract_mfcc(file)
        # Assign label based on file path (folder name)
        label = "PD" if "pd" in file.lower() or "patolog" in file.lower() else "HC"
        data.append([file, label] + features.tolist())
    except Exception as e:
        print(f"⚠️ Skipped: {file} ({e})")

# Save to CSV
df = pd.DataFrame(data, columns=["file_path", "label"] + [f"mfcc_{i+1}" for i in range(13)])
df.to_csv(output_csv, index=False)
print(f"✅ Extracted {len(df)} samples → {output_csv}")
