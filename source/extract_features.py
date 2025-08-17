import os
import librosa
import numpy as np
import pandas as pd

# Path to the main dataset folder (PC-GITA_16kHz)
DATA_DIR = DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/PC-GITA_16kHz")



def list_all_audio_files():
    """
    Recursively search for all .wav files inside DATA_DIR.
    Returns a list of full paths to each audio file.
    """
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):  # Walk through all subfolders and files
        for file in files:
            if file.lower().endswith(".wav"):  # Check only for .wav files
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    return all_files


def extract_mfcc(file_path, n_mfcc=13):
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) features from an audio file.
    - file_path: path to the audio file
    - n_mfcc: number of MFCC coefficients to extract (default = 13)

    Returns the mean value of each MFCC coefficient.
    """
    y, sr = librosa.load(file_path, sr=16000)  # resampling at 16kHz

    mfcc = librosa.feature.mfcc(  # extracting MFCC
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=400,
        hop_length=160,
        win_length=400
    )
    mfcc_mean = np.mean(mfcc, axis=1)  # Compute the mean across time frames
    return mfcc_mean


def process_all_audio(audio_files, output_csv):
    """
    Process all audio files:
    - Extract MFCC features
    - Detect labels (PD vs HC)
    - Save the results into a CSV file
    """
    data = []
    for file_path in audio_files:
        features = extract_mfcc(file_path)

        # Automatically assign labels based on the folder name
        label = "PD" if "pd" in file_path.lower() or "patologica" in file_path.lower() else "HC"

        # Append data: [file_path, label, mfcc_1, mfcc_2, ..., mfcc_13]
        data.append([file_path, label] + features.tolist())

    # Create DataFrame and save it as CSV
    columns = ["file_path", "label"] + [f"mfcc_{i + 1}" for i in range(13)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(data)} features to {output_csv}")


if __name__ == "__main__":
    # Step 1: Search for all audio files
    print("Searching for all .wav files in PC-GITA_16kHz ...")
    audio_files = list_all_audio_files()
    print(f"Found {len(audio_files)} audio files.")

    # Step 2: Extract features and save them to a CSV
    process_all_audio(audio_files, "all_mfcc_features.csv")
