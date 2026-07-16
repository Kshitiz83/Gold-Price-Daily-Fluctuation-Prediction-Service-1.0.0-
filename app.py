import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Body

# Initialize application
app = FastAPI(
    title="Gold Price Daily Fluctuation Prediction Service",
    description="Flexible production API predicting daily gold price differences.",
    version="1.0.0"
)

# Global model artifact placeholder
model = None
model_features = None

@app.on_event("startup")
def load_model_artifact():
    """Mounts the pipeline and extracts the exact feature names seen at fit time."""
    global model, model_features
    model_path = "gold_rf_stationary_model.pkl"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Critical Error: Asset file '{model_path}' not found on local disk.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Dynamically extract feature names from the pipeline's internal feature tracker
    try:
        model_features = model.feature_names_in_.tolist()
    except AttributeError:
        raise AttributeError("Could not extract feature names from pipeline tracker.")

# Complete realistic market snapshot dictionary to serve as your UI example
swagger_sample_payload = {
    "EG_open": 9.35, "EG_high": 9.60, "EG_low": 9.15, "EG_close": 9.45, "EG_volume": 136700.0,
    "PLD_Price": -0.12, "EU_Price": 0.01, "SF_Price": -0.03, "DJ_volume": 0.15, "USB_Price": 0.02,
    "SF_Volume": -0.05, "USO_Volume": 0.11, "USDI_Trend": 0.0, "EU_Trend": 1.0,
    "Year": 2018, "Month": 12, "Day": 31,
    "Gold_Price_Lag1": 120.50, "Gold_Price_Lag2": 110.10, "Gold_Rolling_Mean5": 120.35,
    "Gold_Diff_Lag1": 0.40, "Gold_Diff_Lag2": -0.15
}

@app.post("/predict", tags=["Inference"])
async def predict_gold_movement(
    payload: dict = Body(..., example=swagger_sample_payload)
):
    """
    Ingests market metrics, dynamically matches and orders keys against
    the model feature matrix schema, and executes inference.
    """
    if model is None or model_features is None:
        raise HTTPException(status_code=503, detail="Model artifact is currently unavailable.")
    
    try:
        # Step 1: Standardize user inputs by turning potential underscore variations into space layout strings
        normalized_payload = {}
        for k, v in payload.items():
            new_key = (k.replace("_open", " open")
                        .replace("_high", " high")
                        .replace("_low", " low")
                        .replace("_close", " close")
                        .replace("_volume", " volume")
                        .replace("_Price", " Price")
                        .replace("_Trend", " Trend")
                        .replace("_Adjclose", " Adjclose"))
            normalized_payload[new_key] = v

        # Step 2: Build a clean dictionary mapped precisely to the 24 required model features
        inference_dict = {}
        for feature in model_features:
            inference_dict[feature] = normalized_payload.get(feature, 0.0)
            
        # Step 3: Construct single-row pandas DataFrame with exact training column order
        df = pd.DataFrame([inference_dict])[model_features]
        
        # Step 4: Execute Scikit-Learn production pipeline inference
        prediction = model.predict(df)
        
        return {
            "status": "success",
            "predicted_daily_change_usd": float(round(prediction[0], 4)),
            "direction": "UP" if prediction[0] >= 0 else "DOWN"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Inference Failure: {str(e)}")

@app.get("/health", tags=["System"])
async def health_check():
    """System heartbeat route verifying model health status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }
