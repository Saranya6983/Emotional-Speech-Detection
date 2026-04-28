import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import joblib
import pyttsx3
import threading

# Load model + scaler
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎤 Speech Emotion Detection")

# Record audio
def record_audio(duration=5, fs=22050):
    st.write("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten()

# Feature extraction
def extract_features(audio, sr=22050):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ])

#  Voice function
def speak(msg):
    engine = pyttsx3.init()
    engine.say(msg)
    engine.runAndWait()

# Button click
if st.button("🎤 Start Recording"):

    # Record audio
    audio = record_audio()

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Extract features
    features = extract_features(audio)

    # Apply scaler
    features = scaler.transform([features])

    # Predict
    prediction = model.predict(features)[0]

    # Emotion-based messages
    if prediction == "angry":
        message = "Stay calm 😌 Take a deep breath."
    elif prediction == "sad":
        message = "Everything will be okay 💙 Stay strong."
    elif prediction == "happy":
        message = "Keep smiling 😊 That's great!"
    else:
        message = "Have a nice day 😄"

    # Show output
    st.success(f"Detected Emotion: {prediction}")
    st.info(message)

    #  Voice output (INSIDE button block)
    threading.Thread(target=speak, args=(message,)).start()