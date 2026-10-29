from model_service import FEATURES, predict_from_features

# Synthetic test values in the same feature order used during training
test_features = [
    0.28,   # EAR_mean
    0.04,   # EAR_std
    5.0,    # EAR_low_percent
    0.15,   # MAR_mean
    0.45,   # MAR_max
    0.30,   # MAR_q90
    0.02,   # dEAR_std
    0.03    # dMAR_std
]

result = predict_from_features(test_features)

print("Features:", FEATURES)
print("Prediction:", result["status"])
print("Class:", result["prediction"])
print("Probabilities:", result["probabilities"])