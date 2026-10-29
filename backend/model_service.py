
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "compact_RF_final.joblib"
)

FEATURES = [
    "EAR_mean",
    "EAR_std",
    "EAR_low_percent",
    "MAR_mean",
    "MAR_max",
    "MAR_q90",
    "dEAR_std",
    "dMAR_std",
]

_model = None


def load_model():
    """Load the trained Random Forest pipeline once."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_from_features(feature_values):
    """Predict Alert or Drowsy from eight extracted features."""
    if len(feature_values) != len(FEATURES):
        raise ValueError(
            f"Expected {len(FEATURES)} features, "
            f"received {len(feature_values)}."
        )

    values = np.asarray(feature_values, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("All feature values must be finite.")

    input_data = pd.DataFrame([values], columns=FEATURES)

    model = load_model()
    
    probabilities = model.predict_proba(input_data)[0]

    # Probability of the Drowsy class (class 1)
    drowsy_probability = float(
        probabilities[list(model.classes_).index(1)]
    )


    # Start with a conservative threshold.
    DROWSY_THRESHOLD = 0.68

    prediction = int(drowsy_probability >= DROWSY_THRESHOLD)


    return {
        "prediction": prediction,
        "status": "Drowsy" if prediction == 1 else "Alert",
        "probabilities": {
            str(int(label)): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }
