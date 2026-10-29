
from temporal_features import TemporalFeatureExtractor, FEATURES
from model_service import predict_from_features

extractor = TemporalFeatureExtractor()

# Synthetic measurements for integration testing only.
for i in range(60):
    ear = 0.28 if i % 10 else 0.15
    mar = 0.12 if i % 8 else 0.25
    extractor.add_sample(ear, mar)

if not extractor.ready():
    raise RuntimeError("Temporal feature window is not ready.")

feature_dict = extractor.get_features()

# Preserve the exact feature order expected by the model.
feature_values = [
    feature_dict[name] for name in FEATURES
]

result = predict_from_features(feature_values)

print("Feature aggregation: SUCCESS")
print("Model inference: SUCCESS")
print("Feature count:", len(feature_values))
print("Prediction:", result["status"])
print("Probabilities:", result["probabilities"])
