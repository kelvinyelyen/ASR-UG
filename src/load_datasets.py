import os
import wget
import zipfile
import numpy as np
import librosa


# Function to download and extract the dataset
def download_and_extract_dataset(dataset_link, extract_to):
    # Download the dataset zip file
    dataset_zip_url = dataset_link + "&dl=1"
    dataset_zip_path = os.path.join(extract_to, "dataset.zip")
    wget.download(dataset_zip_url, out=dataset_zip_path)
    print("Dataset downloaded successfully.")

    # Extract the dataset
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Dataset extracted successfully.")


# Function to load and preprocess data
def load_and_preprocess_data(data_dir):
    labels = []
    audio_paths = []

    # Traverse the directory structure to extract labels and audio paths
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for audio_file in os.listdir(label_dir):
                audio_path = os.path.join(label_dir, audio_file)
                audio_paths.append(audio_path)
                labels.append(label)

    print("Number of samples:", len(audio_paths))
    return labels, audio_paths


# Function to preprocess audio data (MFCC extraction)
def preprocess_audio(audio_path, sequence_length):
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # Pad or truncate MFCCs to match the sequence length
    if mfccs.shape[1] < sequence_length:
        pad_width = sequence_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :sequence_length]
    return mfccs
