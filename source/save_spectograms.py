import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# === Input root: folderul care conține toate propozițiile ===
input_base = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\data\PC-GITA_16kHz\sentences2"


# === Output directory for saving spectrograms ===
output_root = "source/reports/spectrograms"
classes = ["hc", "pd"]
sets = ["training", "validation"]

# === Create output folder structure ===
for set_name in sets:
    for cls in classes:
        os.makedirs(os.path.join(output_root, set_name, cls.upper()), exist_ok=True)

# === Colectează toate fișierele .wav din TOATE propozițiile ===
all_files = []
labels = []

for sentence_folder in os.listdir(input_base):
    sentence_path = os.path.join(input_base, sentence_folder, "non-normalized")
    for cls in classes:
        class_path = os.path.join(sentence_path, cls)
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(".wav"):
                all_files.append(os.path.join(class_path, file))
                labels.append(cls)

# === Split în train/val ===
train_files, val_files, y_train, y_val = train_test_split(
    all_files, labels, test_size=0.2, stratify=labels, random_state=42
)

file_split_map = {f: ("training", y) for f, y in zip(train_files, y_train)}
file_split_map.update({f: ("validation", y) for f, y in zip(val_files, y_val)})

# === Funcție de salvare a spectogramelor ===
def save_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)

        S = librosa.feature.melspectrogram(
            y=y,
            sr=16000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=64
        )

        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_dB, sr=sr)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        print(f"⚠️ Error with {file_path}: {e}")
        return False

# === Generate spectograms ===
count = 0
for file_path, (set_type, label) in file_split_map.items():
    label_out = label.upper()
    filename = Path(file_path).stem + ".png"
    out_path = os.path.join(output_root, set_type, label_out, filename)

    if save_spectrogram(file_path, out_path):
        count += 1

print(f"\n✅ Total spectrograms saved: {count}")
print(f"Saved in: {output_root}")
