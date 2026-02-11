import joblib
import numpy as np
from fastapi import FastAPI
from app.schema import HeartDiseaseInput
from src.config import MODEL_PATH

app = FastAPI(title="Heart Disease Prediction API")

# Load model directly (NOT dictionary)
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    try:
        input_data = np.array([[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        return {"error": str(e)}
