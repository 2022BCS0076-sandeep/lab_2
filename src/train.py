import json
import pickle
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# LOAD EXPERIMENT CONFIG
# -----------------------
config_path = sys.argv[1]
config = json.loads(open(config_path).read())

# -----------------------
# LOAD DATASET
# -----------------------
df = pd.read_csv("data/winequality-red.csv", sep=';')
X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------
# TRAIN/TEST SPLIT
# -----------------------
test_size = config["split"]

# Stratified or normal split
if config.get("stratified", False):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# -----------------------
# MODEL SELECTION
# -----------------------
model_name = config["model"]
hyperparams = config.get("hyperparameters", {})
preprocessing = config.get("preprocessing", "None")

if model_name == "LinearRegression":
    model = LinearRegression(**hyperparams)

elif model_name == "Ridge":
    model = Ridge(**hyperparams)

elif model_name == "RandomForest":
    model = RandomForestRegressor(**hyperparams, random_state=42)

else:
    raise ValueError(f"Unknown model: {model_name}")

# -----------------------
# PIPELINE (if needed)
# -----------------------
if preprocessing == "Standardization":
    model = make_pipeline(StandardScaler(), model)

# -----------------------
# TRAIN & PREDICT
# -----------------------
model.fit(X_train, y_train)
preds = model.predict(X_test)

# -----------------------
# METRICS
# -----------------------
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Print cleanly for GitHub Actions summary
print("Experiment:", config["id"])
print("Model:", model_name)
print("MSE:", mse)
print("R2:", r2)

# -----------------------
# SAVE OUTPUTS
# -----------------------
result = {
    "experiment_id": config["id"],
    "model": model_name,
    "hyperparameters": hyperparams,
    "preprocessing": preprocessing,
    "feature_selection": config.get("feature_selection", "None"),
    "split": test_size,
    "stratified": config.get("stratified", False),
    "mse": mse,
    "r2_score": r2
}

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

with open(f"outputs/models/{config['id']}.pkl", "wb") as f:
    pickle.dump(model, f)

with open(f"outputs/results/{config['id']}.json", "w") as f:
    json.dump(result, f, indent=4)
