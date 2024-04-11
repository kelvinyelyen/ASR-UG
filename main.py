from load_datasets import load_data, preprocess_data, load_data_with_tf
from model import build_model
from train import train_model
from evaluate import evaluate_model


def main():
    data_dir = "./data/mini_speech_commands"
    audio_paths, transcripts = load_data(data_dir)
    print("Number of audio files:", len(audio_paths))
    print("Number of transcripts:", len(transcripts))

    # Set your desired sequence length here
    sequence_length = 1000
    features, labels = preprocess_data(
        audio_paths, transcripts, sequence_length)
    print("Number of features after preprocessing:", len(features))
    print("Number of labels after preprocessing:", len(labels))
    # Print shape of first feature for verification
    print("Features shape:", features[0].shape)
    print("Labels:", labels)  # Print labels for verification

    # Load data using TensorFlow
    training_set, validation_set = load_data_with_tf(data_dir)

    # Build the model
    input_shape = features[0].shape
    output_dim = len(training_set.class_names)
    model = build_model(input_shape=input_shape, output_dim=output_dim)

    # Train the model
    train_model(model, training_set, validation_set)

    # Evaluate the model
    evaluate_model(model, validation_set)


if __name__ == "__main__":
    main()
