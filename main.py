import streamlit as st
import os
import keras
import librosa  # Pour l'extraction des features et la lecture des fichiers wav
import librosa.display  # Pour récupérer les spectrogrammes des audio
import librosa.feature
import numpy as np

model = keras.models.load_model("./neuralNetwork.hdf5")
import joblib
scaler = joblib.load("scaler.pkl")

genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def audio_pipeline(audio):
    features = []

    # Calcul du ZCR

    zcr = librosa.zero_crossings(audio)
    features.append(sum(zcr))

    # Calcul de la moyenne du Spectral centroid

    spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
    features.append(np.mean(spectral_centroids))

    # Calcul du spectral rolloff point

    rolloff = librosa.feature.spectral_rolloff(y=audio)
    features.append(np.mean(rolloff))

    # Calcul des moyennes des MFCC

    mfcc = librosa.feature.mfcc(y=audio)

    for x in mfcc:
        features.append(np.mean(x))

    return features


# Load and preprocess the audio file
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=None)

    # Apply the same feature extraction and scaling as you did during training
    features = audio_pipeline(audio)
    scaled_features = scaler.transform([features])

    return scaled_features


# Make predictions on the preprocessed audio
def predict_top_genres(file_path, top_n=3):
    scaled_features = preprocess_audio(file_path)
    predicted_probabilities = model.predict(scaled_features)
    top_n_indices = np.argsort(predicted_probabilities[0])[::-1][:top_n]
    top_n_genres = [genres[idx] for idx in top_n_indices]
    prediction_percentages = [
        predicted_probabilities[0][idx] * 100 for idx in top_n_indices
    ]
    return top_n_genres, prediction_percentages


# Provide the path to the audio file you want to classify
audio_file_path = "./dataset/to_check/sound/hiphop/hiphop.00000.wav"
top_genres, prediction_percentages = predict_top_genres(audio_file_path)

for i, genre in enumerate(top_genres):
    print(i + 1, genre, "{:.2f}%".format(prediction_percentages[i]))


def save_uploaded_file(uploaded_file):
    # Specify the directory to save the uploaded files
    save_directory = "uploads"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create a file path based on the original file name
    file_path = os.path.join(save_directory, "toAnalyse.wav")

    # Write the file contents to the specified file path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File saved as {file_path}")
    return save_directory + "/toAnalyse.wav"


def main():
    st.title("Audio File Uploader")

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if audio_file is not None:
        path = save_uploaded_file(audio_file)
        st.audio(audio_file, format="audio/mp3")
        top_genres, prediction_percentages = predict_top_genres(path)

        for i, genre in enumerate(top_genres):
            st.text(f"{genre} : " + "{:.2f}%".format(prediction_percentages[i]))
            print(i + 1, genre, "{:.2f}%".format(prediction_percentages[i]))


if __name__ == "__main__":
    main()
