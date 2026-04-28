import sounddevice as sd
import numpy as np
import librosa
import joblib

# Load model + scaler
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

# Record audio
def record_audio(duration=5, fs=22050):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten()

# Extract features
def extract_features_live(audio, sr=22050):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    return np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0)
    ])

# Run
audio = record_audio()

# Normalize audio
audio = audio / np.max(np.abs(audio))

features = extract_features_live(audio)
features = scaler.transform([features])

prediction = model.predict(features)

emotion = prediction[0]

if emotion == "angry":
    message = "Stay calm 😌 Take a deep breath."
elif emotion == "sad":
    message = "Everything will be okay 💙 Stay strong."
elif emotion == "happy":
    message = "Keep smiling 😊 That's great!"
else:
    message = "Have a nice day 😄"

print("Detected Emotion:", emotion)
print("Message:", message)
