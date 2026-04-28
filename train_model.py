import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from utils import extract_features

# Data storage
X = []
y = []

dataset_path = "dataset/"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

# Read dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if ".wav" in file.lower():
            parts = file.split("-")

            if len(parts) > 2:
                emotion_code = parts[2]

                if emotion_code in emotion_map:
                    file_path = os.path.join(root, file)

                    print("Processing:", file_path)

                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion_map[emotion_code])

# CHECK DATA
print("Total samples:", len(X))

#  STOP if empty
if len(X) == 0:
    print("ERROR: No data found! Check dataset folder.")
    exit()

# Convert to numpy
X = np.array(X)
y = np.array(y)

# 🔥 SCALE DATA
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save
joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")