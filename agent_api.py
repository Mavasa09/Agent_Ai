from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load your model
model = joblib.load('municipal_ai_model.pkl')

# Define input schema (adjust fields to match your features!)
class InputData(BaseModel):
    feature_1: float  # Replace with your actual feature names
    feature_2: float
    # ... add all features used in your notebook

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame (like your notebook's preprocessing)
    input_df = pd.DataFrame([data.dict()])
    
    # Predict (ensure input matches training data format)
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction), "action": "take_action_based_on_prediction"}