import os
import numpy as np
import librosa
import wget
import zipfile
import tensorflow as tf

BOLD = '\033[1m'
GREEN = '\033[92m'
RESET = '\033[0m'


def download_and_extract_dataset(dataset_link, extract_to):
    if os.listdir(extract_to):
        print(
            f"{BOLD}{GREEN}Dataset directory is not empty. Skipping download and extraction.{RESET}")
        return
    print(f"{BOLD}Downloading datasets...{RESET}")
    dataset_zip_file = wget.download(dataset_link)
    print(f"{BOLD}Extracting...{RESET}")
    with zipfile.ZipFile(dataset_zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(dataset_zip_file)


def load_data(data_dir):
    audio_paths, transcripts = [], []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            print("Class directory:", class_path)
            for audio_file in os.listdir(class_path):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(class_path, audio_file)
                    transcript_path = os.path.join(
                        class_path, audio_file.replace(".wav", ".txt"))
                    print("Audio path:", audio_path)
                    print("Transcript path:", transcript_path)
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            transcript = f.read().strip()
                        audio_paths.append(audio_path)
                        transcripts.append(transcript)
                    except FileNotFoundError:
                        print("Warning: Transcript file not found for:", audio_path)
    return audio_paths, transcripts


def preprocess_data(audio_paths, transcripts, sequence_length):
    features, labels = [], []
    for audio_path, transcript in zip(audio_paths, transcripts):
        print("Processing audio file:", audio_path)
        print("Transcript:", transcript)
        audio, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.append(mfccs.T)
        label = [ord(char) - ord('a') for char in transcript.lower()]
        labels.append(label)
    return features, labels


def load_data_with_tf(data_dir, batch_size=16, validation_split=0.2, output_sequence_length=16000):
    training_set = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        output_sequence_length=output_sequence_length,
        seed=0,
        subset='training'
    )
    validation_set = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        output_sequence_length=output_sequence_length,
        seed=0,
        subset='validation'
    )
    return training_set, validation_set


dataset_link = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
extract_to = "./data"
data_dir = "./data/mini_speech_commands"

# Download and extract the dataset
download_and_extract_dataset(dataset_link, extract_to)

# Load data using TensorFlow
training_set, validation_set = load_data_with_tf(data_dir)

label_names = np.array(training_set.class_names)
print("Label names:", label_names)

# Preprocess data
# Further processing of training_set and validation_set as needed

for batch in training_set.take(1):  # Only take one batch
    print("Features shape:", batch[0].shape)
    print("Labels shape:", batch[1].shape)
