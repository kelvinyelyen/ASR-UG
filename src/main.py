import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from load_datasets import download_and_extract_dataset, load_and_preprocess_data, preprocess_audio
from model import build_model
from train import train_model
from evaluate import evaluate_model


def main():
    dataset_url = "https://www.dropbox.com/scl/fo/jvcx6dwpvuwaiboijg34d/ALVdJuoj1IyybQJ2SC3thHc?rlkey=px94zhss4kr66c619q1jfqwzt&st=9jfmfgun"
    extract_to = "data"
    data_dir = "data"
    sequence_length = 1000

    # Download and extract the dataset
    download_and_extract_dataset(dataset_url, extract_to)

    # Preprocess audio data and labels
    labels, audio_paths = load_and_preprocess_data(extract_to)

    # Process features
    features = []
    for audio_path in audio_paths:
        try:
            feature = preprocess_audio(audio_path, sequence_length)
            features.append(feature)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    features = np.array(features)
    print("Number of samples processed:", len(features))

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Build the model
    input_shape = features[0].shape
    num_classes = labels.shape[1]
    model = build_model(input_shape=input_shape, output_dim=num_classes)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, X_test, y_test)

    # Visualize training/validation loss and accuracy
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
