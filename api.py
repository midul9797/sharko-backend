from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import sys
import os
import pandas as pd

# Add the current directory to Python path to import from index.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function from index.py
from index import batch_predict, load_models, predict_shark_habitat,predict_shark_presence
# Initialize FastAPI app
app = FastAPI(title="Shark Prediction API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):  # Default file
    points_per_polygon: int = 200
    prediction_date: str = "2030-05-14"
    point_generation_method: str = "adaptive"  # 'random', 'grid', or 'adaptive'
    epsilon: float = 1.0  # DBSCAN parameter
    min_samples: int = 5  # DBSCAN parameter

class PredictionResponse(BaseModel):
    success: bool
    message: str
    presence_geojson_data: Optional[dict] = None
    habitat_geojson_data: Optional[dict] = None
    prediction_stats: Optional[dict] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Shark Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        models = load_models()
        return {
            "status": "healthy", 
            "models_loaded": list(models.keys()),
            "available_models": ["sst", "ssh", "chla", "shark"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {str(e)}")

@app.get("/predict/presence")
async def shark_presence(date:str):
    """
    Main prediction endpoint that calls the main function from index.py
    and returns GeoJSON feature collection for frontend visualization
    """
    try:
        # Call the main function from index.py
        geojson_file = "initial_zone_1_epsilon.geojson"
        points_per_polygon: int = 200
        point_generation_method: str = "adaptive"  # 'random', 'grid', or 'adaptive'
        epsilon: float = 2.0  # DBSCAN parameter
        min_samples: int = 4  # DBSCAN parameter
        geojson_data = predict_shark_presence(
            geojson_file=geojson_file,
            points_per_polygon=points_per_polygon,
            prediction_date=date,
            point_generation_method=point_generation_method,
            epsilon=epsilon,
            min_samples=min_samples)
        if geojson_data:
            # Handle tuple return (presence, habitat) or single return
            
            return PredictionResponse(
                success=True,
                message="Clusters found",
                presence_geojson_data=geojson_data,
                prediction_stats=None
            )
            
        else:
            return PredictionResponse(
                success=False,
                message="No valid clusters found. Try adjusting parameters.",
                presence_geojson_data=None,
                habitat_geojson_data=None,
                prediction_stats=None
            )
           
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
@app.get("/predict/habitat")
async def shark_habitat(date:str, shark_name:str):
    """
    Main prediction endpoint that calls the main function from index.py
    and returns GeoJSON feature collection for frontend visualization
    """
    try:
        # Call the main function from index.py
        geojson_file = "initial_zone_1_epsilon.geojson"
        points_per_polygon: int = 200
        point_generation_method: str = "adaptive"  # 'random', 'grid', or 'adaptive'
        epsilon: float = 2.0  # DBSCAN parameter
        min_samples: int = 5  # DBSCAN parameter
        
        geojson_data = predict_shark_habitat(
            geojson_file=geojson_file,
            points_per_polygon=points_per_polygon,
            prediction_date=date,
            point_generation_method=point_generation_method,
            epsilon=epsilon,
            min_samples=min_samples,
            shark_name=shark_name)
        
        if geojson_data:
            # Handle tuple return (presence, habitat) or single return
            
            return PredictionResponse(
                success=True,
                message="No valid clusters found. Try adjusting parameters.",
                presence_geojson_data=None,
                habitat_geojson_data=geojson_data,
                prediction_stats=None
            )
        else:
            return PredictionResponse(
                success=False,
                message="No valid clusters found. Try adjusting parameters.",
                presence_geojson_data=None,
                habitat_geojson_data=None,
                prediction_stats=None
            )
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/location")
async def location_predict(lat: float, lon: float, date: str = "2030-05-14"):
    """
    Simple prediction endpoint for single point prediction
    """
    try:
        models = load_models()
        
        # Create a simple prediction point
        prediction_points = pd.DataFrame({
            'lat': [lat],
            'lon': [lon],
            'date': [date]
        })
        
        # Run prediction
        results = batch_predict(prediction_points, models)
        
        # Return single prediction result
        result = results.iloc[0]
        
        return {
            "coordinates": {"lat": lat, "lon": lon},
            "date": date,
            "predictions": {
                "shark_presence": float(result['predicted_shark_presence']),
                "sst": float(result['predicted_sst']),
                "ssh": float(result['predicted_ssh']),
                "chla": float(result['predicted_chla'])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simple prediction failed: {str(e)}")
class Question(BaseModel):
    context: list[str]
    question: str

# Global variable for lazy loading
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        try:
            rag_chain = build_rag_chain()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"RAG chain initialization failed: {str(e)}")
    return rag_chain


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
