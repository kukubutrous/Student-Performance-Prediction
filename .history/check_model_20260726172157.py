import joblib
from pathlib import Path

MODEL_DIR = Path("models")

model = joblib.load(MODEL_DIR / "best_model.pkl")

print(type(model))

print()

print("Number of features:")
print(model.n_features_in_)

print()

print("Feature names:")
print(model.feature_names_in_)