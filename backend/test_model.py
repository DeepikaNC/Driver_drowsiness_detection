
from model_service import FEATURES, load_model

model = load_model()

print("Model loaded successfully!")
print("Model type:", type(model).__name__)
print("Expected features:", FEATURES)
print("Model classes:", model.classes_)

print("\nFeature count:", len(FEATURES))
print("Model input count:", model.n_features_in_)
