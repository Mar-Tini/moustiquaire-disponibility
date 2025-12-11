import gdown
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_ID = "1gp6EOOc39eg7UCCFs7RN2yUZc53SnYf7"
FEATURE_ID = "1gjBqGVYpixEd_vJKR3xiFg9lKeE-uBHT"

def load_model_and_predict(X, y_true):


    os.makedirs("models", exist_ok=True)

 
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", "models/model.pkl", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={FEATURE_ID}", "models/features.pkl", quiet=False)


    model = joblib.load("models/model.pkl")
    selected_features = joblib.load("models/features.pkl")

    X = X[selected_features]


    y_prob = model.predict_proba(X)[:, 0]
    y_true_binary = (y_true == "NON OBSERVEE").astype(int)

    thresholds = np.linspace(0, 1, 101)
    f1_scores = [
        f1_score(y_true_binary, (y_prob >= t).astype(int))
        for t in thresholds
    ]

    optimal_threshold = thresholds[np.argmax(f1_scores)]

    return model, X, y_prob, optimal_threshold
