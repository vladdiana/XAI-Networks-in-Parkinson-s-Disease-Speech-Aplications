import os, librosa, librosa.display, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# === INPUT: ONLY the 'monologue' folder ===
MONO_ROOT = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\data\PC-GITA_16kHz\monologue"

# === OUTPUT ===
OUT_ROOT  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms"
os.makedirs(OUT_ROOT, exist_ok=True)

def infer_label_from_path(p: str):
    parts = [q.lower() for q in Path(p).parts]
    for q in parts:
        if q.startswith("hc"):
            return "HC"
        if q.startswith("pd") or "patolog" in q:
            return "PD"
    return None

def collect_wavs(root):
    files, labels = [], []
    for dp, _, fnames in os.walk(root):
        for f in fnames:
            if not f.lower().endswith(".wav"):
                continue
            full = os.path.join(dp, f)
            lab = infer_label_from_path(full)
            if lab is None:
                continue
            files.append(full); labels.append(lab)
    return files, labels

def save_melspec(wav_path, out_png):
    y, sr = librosa.load(wav_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, win_length=400, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3,3))
    librosa.display.specshow(S_dB, sr=sr)
    plt.axis("off"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

print("Scanning WAVs in:", MONO_ROOT)
files, labels = collect_wavs(MONO_ROOT)
print(f"Found {len(files)} labeled WAVs in 'monologue'.")

if len(files) == 0:
    raise RuntimeError("No WAVs found in 'monologue'. Check the path and HC/PD folder names.")

tr_f, va_f, y_tr, y_va = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

def dump_split(fs, lbs, split):
    count=0
    for fp,lab in zip(fs,lbs):
        out_png = os.path.join(OUT_ROOT, split, lab, Path(fp).stem + ".png")
        try:
            save_melspec(fp, out_png); count+=1
        except Exception as e:
            print(f"Skip {fp}: {e}")
    print(f"Saved {count} spectrograms to: {os.path.join(OUT_ROOT, split)}")

dump_split(tr_f, y_tr, "training")
dump_split(va_f, y_va, "validation")
print(f"✅ Done. Output: {OUT_ROOT}")
