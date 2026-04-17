import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import logging
from fastapi import FastAPI
import numpy as np
from core.adaptive_model import AdaptiveAutoencoder
from intelligence.counterfactual_engine import CounterfactualEngine

logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize model without loading (load on demand)
model = AdaptiveAutoencoder(input_dim=5)
model_loaded = False

# Try to load existing model, but don't fail if it doesn't exist yet
try:
    model_files = [
        os.path.exists("models/model.keras"),
        os.path.exists("models/model.h5"),
        os.path.exists("models/scaler.pkl")
    ]
    
    if any(model_files):
        model.load()
        model_loaded = True
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model files not found - API will use untrained model")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.warning("Continuing with untrained model")

# Initialize CounterfactualEngine
counterfactual = CounterfactualEngine(model)


@app.get("/")
def home():
    return {
        "status": "API running",
        "model_loaded": model_loaded,
        "endpoints": {
            "/health": "Health check",
            "/predict": "POST - Model prediction",
            "/status": "Model status"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/status")
def model_status():
    """Get current model status"""
    return {
        "model_loaded": model_loaded,
        "model_type": "AdaptiveAutoencoder",
        "input_dim": 5,
        "message": "Model loaded from disk" if model_loaded else "Using untrained model"
    }


@app.post("/predict")
def predict(data: list):
    """
    Model prediction endpoint
    
    Expected JSON:
    {
        "data": [feature1, feature2, feature3, feature4, feature5]
    }
    """
    try:
        sample = np.array(data).reshape(1, -1)
        
        # Validate input dimensions
        if sample.shape[1] != 5:
            return {
                "error": f"Expected 5 features, got {sample.shape[1]}",
                "status": "ERROR"
            }, 400
        
        error, uncertainty = model.predict_with_uncertainty(sample)
        status = get_status(error)
        
        # Generate counterfactual analysis if WARNING or higher
        cf_results = []
        boundaries = []
        if error > 0.02:
            cf_results = counterfactual.generate(sample[0])
            boundaries = counterfactual.find_failure_boundary(sample[0])

        return {
            "error": float(error),
            "uncertainty": float(uncertainty),
            "status": status,
            "model_used": "trained" if model_loaded else "untrained",
            "counterfactuals": cf_results[:5],
            "failure_boundaries": boundaries
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e), "status": "ERROR"}, 500


def get_status(error):
    if error < 0.01:
        return "NORMAL"
    elif error < 0.03:
        return "WARNING"
    else:
        return "CRITICAL"


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)