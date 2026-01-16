import pandas as pd
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Load dataset
data_path = "data/winequality-red.csv"
df = pd.read_csv(data_path, sep=';')

# Split features & target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & metrics
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Print to console
print("MSE:", mse)
print("R2:", r2)

# Save model as .pkl
os.makedirs("outputs/models", exist_ok=True)
with open("outputs/models/model.pkl", "wb") as f:
    pickle.dump(model, f)


# Save metrics
results = {"mse": mse, "r2_score": r2}
os.makedirs("outputs/results", exist_ok=True)

with open("outputs/results/results.json", "w") as f:
    json.dump(results, f, indent=4)
