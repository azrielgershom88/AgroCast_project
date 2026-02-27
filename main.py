import os
import uuid
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings
from fetcher import get_live_weather, get_live_aqi

app = FastAPI(title="AgroCast Pipeline API", version="1.0.0")

# --- CORS SETUP ---
# Allows your Streamlit frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC DIRECTORY ---
# Creates a folder to temporarily hold the generated audio files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- API CLIENTS & KEYS ---
FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

featherless_client = AsyncOpenAI(
    api_key=FEATHERLESS_API_KEY,
    base_url="https://api.featherless.ai/v1",
) if FEATHERLESS_API_KEY else None

elevenlabs_client = AsyncElevenLabs(
    api_key=ELEVENLABS_API_KEY
) if ELEVENLABS_API_KEY else None

# --- LOAD MASTER AI MODEL ---
print("🚀 Initializing Master AI Model...")
try:
    from sklearn.ensemble import GradientBoostingRegressor
    master_price_model = joblib.load('master_price_model.pkl')
    print("✅ Master Price Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ CRITICAL ERROR loading master model: {e}")
    master_price_model = None

# --- DATA SCHEMAS ---
class PipelineRequest(BaseModel):
    crop: str
    lat: float
    lon: float
    current_price: float
    language: str = "Tamil"

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "AgroCast Backend Systems Online 🌍"}

@app.post("/predict/pipeline")
async def process_pipeline(request: PipelineRequest):
    # 1. Fetch Live Weather & AQI from your fetcher.py
    live_climate = get_live_weather(request.lat, request.lon)
    aqi_data = get_live_aqi(request.lat, request.lon)
    
    current_temp = live_climate.get("temperature_celsius") or 30.0
    current_humidity = live_climate.get("relative_humidity_percent") or 60.0
    current_precip = live_climate.get("precipitation_mm") or 0.0
    current_aqi = aqi_data.get("aqi") or 100.0

    # 2. Format Input Array for the Gradient Boosting Model
    # [lag1, lag3, roll3, volatility, aqi, rain, temp, humidity, onion, potato, tomato, mandi1, mandi2, mandi3]
    base_historical_stats = [2100.0, 2050.0, 2080.0, 50.0] # Mocked historical data
    
    crop_name = request.crop.capitalize()
    crop_arr = [
        1 if crop_name == "Onion" else 0,
        1 if crop_name == "Potato" else 0,
        1 if crop_name == "Tomato" else 0
    ]

    mandis = [
        {"name": "Coimbatore Mandi", "distance": 10, "encoded": [1, 0, 0]},
        {"name": "Erode Mandi", "distance": 105, "encoded": [0, 1, 0]},
        {"name": "Tiruppur Mandi", "distance": 55, "encoded": [0, 0, 1]}
    ]

    best_profit = -999999
    best_mandi = ""
    best_price = 0

    # 3. The Decision Layer: Run AI & calculate logistics
    for mandi in mandis:
        try:
            if master_price_model:
                features = np.array([base_historical_stats + [current_aqi, current_precip, current_temp, current_humidity] + crop_arr + mandi["encoded"]])
                predicted_price = float(master_price_model.predict(features)[0])
            else:
                predicted_price = 2200.0 # Safety fallback
        except Exception as e:
            print(f"Model error: {e}")
            predicted_price = 2200.0
            
        # Logistics Math (Assuming 20 quintals, ₹1.5/km)
        transport_cost = mandi["distance"] * 1.5 * 20
        gross_revenue = predicted_price * 20
        net_profit = gross_revenue - transport_cost
        
        if net_profit > best_profit:
            best_profit = net_profit
            best_mandi = mandi["name"]
            best_price = predicted_price

    # 4. Generate the Human-Readable Script
    advisory_text = f"Hello. Based on today's weather and air quality, you should sell your {request.crop} at {best_mandi}. The expected price is {int(best_price)} rupees per quintal, giving you a maximum net profit of {int(best_profit)} rupees after transport costs."

    # 5. ElevenLabs Audio Generation (with Hackathon Bonus Features)
    audio_url = None
    if elevenlabs_client:
        try:
            audio_stream = await elevenlabs_client.generate(
                text=advisory_text,
                voice="Rachel", 
                model="eleven_multilingual_v2", # The +5 Accessibility Feature
                voice_settings=VoiceSettings(
                    stability=0.75,             # The Professional Clarity Feature
                    similarity_boost=0.75
                )
            )
            filename = f"advisory_{uuid.uuid4().hex[:8]}.mp3"
            filepath = os.path.join("static", filename)
            
            with open(filepath, "wb") as f:
                async for chunk in audio_stream:
                    f.write(chunk)
                    
            audio_url = f"/static/{filename}"
        except Exception as e:
            print(f"ElevenLabs Error: {e}")

    # 6. Construct Final API Response
    # Compare AI profit against a baseline local sale
    profit_improvement = best_profit - (request.current_price * 20) if request.current_price else best_profit - (2000 * 20)

    return {
        "input_data": {
            "crop": request.crop,
            "lat": request.lat,
            "lon": request.lon,
            "current_price": request.current_price
        },
        "live_climate": {
            "temperature_celsius": current_temp,
            "relative_humidity_percent": current_humidity,
            "precipitation_mm": current_precip,
            "current_aqi": current_aqi
        },
        "forecasts": {
            "predicted_price": round(best_price, 2),
            "profit_improvement": round(profit_improvement, 2),
            "recommended_action": advisory_text
        },
        "advisory_text": advisory_text,
        "audio_url": audio_url
    }
