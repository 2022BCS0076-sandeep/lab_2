from fastapi import FastAPI
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Inference API for predicting wine quality using trained Random Forest ML model",
    version="1.0"
)

# Load trained model (updated to your repo structure)
model = joblib.load("outputs/model.pkl")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API is running"}

# Prediction endpoint with 11 inputs
@app.post("/predict")
def predict_wine_quality(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float
):
    # Arrange inputs into model format (2D array)
    features = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    # Perform prediction
    prediction = model.predict(features)

    return {
        "name": "Sandeep Chintu",
        "roll_no": "2022BCS0076",
        "predicted_wine_quality": int(prediction[0])
    }
