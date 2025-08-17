import os
import librosa
import numpy as np
import pandas as pd

# Absolute path to the DDK folder in PC-GITA_16kHz
DATA_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\data\PC-GITA_16kHz\DDK analysis"
OUT_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "all_mfcc_features.csv")

def list_wav_files(root):
    files = []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(dp, f))
    return files

def extract_mfcc(path, n_mfcc=13):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length=160, win_length=400)
    return np.mean(mfcc, axis=1)

rows = []
for wav in list_wav_files(DATA_DIR):
    try:
        feats = extract_mfcc(wav)
        label = "PD" if ("pd" in wav.lower() or "patolog" in wav.lower()) else "HC"
        rows.append([wav, label] + feats.tolist())
    except Exception as e:
        print(f"Skipped {wav}: {e}")

df = pd.DataFrame(rows, columns=["file_path", "label"] + [f"mfcc_{i+1}" for i in range(13)])
df.to_csv(OUT_CSV, index=False)
print(f"✅ Extracted {len(df)} samples → {OUT_CSV}")
