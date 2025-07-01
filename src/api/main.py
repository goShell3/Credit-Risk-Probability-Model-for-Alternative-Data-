from fastapi import FastAPI
from src.api.pydantic_models import PredictionInput, PredictionResponse
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load the best model from Mflow
model = mlflow.pyfunc.load_model("models:/best_random_forest/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.dict()])
    prob = model.predict(df)[0]
    return PredictionResponse(risk_probability=prob)
