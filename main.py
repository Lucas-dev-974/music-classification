import streamlit as st
import os
import keras
import librosa  # Pour l'extraction des features et la lecture des fichiers wav
import librosa.display  # Pour récupérer les spectrogrammes des audio
import librosa.feature
import numpy as np
import joblib

model = keras.models.load_model("./models/modelF.hdf5")

scaler = joblib.load("./models/scalerModelF.pkl")

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

    chroma_stft = librosa.feature.chroma_stft(y=audio)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))

    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.var(rms))

    # Calcul de la moyenne du Spectral centroid

    # spectral_centroids = librosa.feature.spectral_centroid(y=audio)[0]
    spectral_centroids = librosa.feature.spectral_centroid(y=audio)
    features.append(np.mean(spectral_centroids))
    features.append(np.var(spectral_centroids))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)
    features.append(np.mean(spectral_bandwidth))
    features.append(np.var(spectral_bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=audio)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))

    harmony = librosa.effects.harmonic(y=audio)
    features.append(np.mean(harmony))
    features.append(np.var(harmony))

    tempo = librosa.feature.tempo(y=audio)
    features.append(tempo[0])

    # Calcul des moyennes des MFCC

    mfcc = librosa.feature.mfcc(y=audio)

    for x in mfcc:
        features.append(np.mean(x))
        features.append(np.var(x))

    return features


# Load and preprocess the audio file
def preprocess_audio(file_path):
    # -------------------------------
    audio, _ = librosa.load(file_path, sr=None)

    # Apply the same feature extraction and scaling as you did during training
    features = audio_pipeline(audio)
    scaled_features = scaler.transform([features])

    return scaled_features


# Make predictions on the preprocessed audio
def predict(file_path, top_n=3):
    # Preprocess audio file
    scaled_features = preprocess_audio(file_path)

    # Make prediction
    predicted_probabilities = model.predict(scaled_features)

    print("brut prediction: ", predicted_probabilities)

    # Get top_n highest
    top_n_index = np.argsort(predicted_probabilities[0])
    print("top_n_indexe: ", top_n_index)

    # For
    top_n_genres = [genres[idx] for idx in top_n_index]
    print("top n genres:", top_n_genres)

    prediction_percentages = [
        predicted_probabilities[0][idx] * 100 for idx in top_n_index
    ]
    return top_n_genres, prediction_percentages


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


# Main streamlit app
def main():
    # Show information like title, audio input uploader, audio control with
    # st.<component>
    st.title("Audio File Uploader")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if audio_file is not None:
        # Save file in uploads folder
        path = save_uploaded_file(audio_file)
        st.audio(audio_file, format="audio/mp3")

        # Make prediction, get genres 3 first genres recognised
        genres, prediction_percentages = predict(path)

        # For each genres show text with its percet %
        for i, genre in enumerate(genres):
            st.text(f"{genre} : " + "{:.2f}%".format(prediction_percentages[i]))
            print(i + 1, genre, "{:.2f}%".format(prediction_percentages[i]))


if __name__ == "__main__":
    main()
