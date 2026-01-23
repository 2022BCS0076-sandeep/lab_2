from fastapi import FastAPI
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Inference API for predicting wine quality using trained Random Forest ML model",
    version="1.0"
)

# Load trained model (from training pipeline)
model = joblib.load("output/model.pkl")

# Health check
@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API is running"}

# Prediction endpoint
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

    prediction = model.predict(features)

    return {
        "name": "Karri Lakshmi Narasimha Reddy",
        "roll_no": "2022BCS0028",
        "predicted_wine_quality": int(prediction[0])
    }