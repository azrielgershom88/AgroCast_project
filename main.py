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
from elevenlabs import save
from fetcher import get_live_weather, get_live_aqi

app = FastAPI(title="AgroCast Pipeline API")

# Setup CORS for Frontend Team (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve audio files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Variables and Clients
FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY")
FEATHERLESS_API_URL = "https://api.featherless.ai/v1"
FEATHERLESS_MODEL = "deepseek-ai/DeepSeek-V3-0324" 

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

featherless_client = AsyncOpenAI(
    api_key=FEATHERLESS_API_KEY,
    base_url=FEATHERLESS_API_URL,
)
elevenlabs_client = AsyncElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

# Load ML Models cleanly at startup
print("Loading ML models...")
try:
    environmental_model = joblib.load("environmental_model.pkl")
    price_model = joblib.load("price_model.pkl")
    print("Successfully loaded ML models.")
except FileNotFoundError as e:
    print(f"Warning: Model not found. Did you run create_mock_models.py? ({e})")
    environmental_model = None
    price_model = None

class PredictionRequest(BaseModel):
    lat: float
    lon: float
    crop: str
    yield_amount: float
    current_price: float
    distant_market_price: float
    transport_cost: float
    language: str = "Tanglish"
    intent: str = "full_advice"

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy", "models_loaded": environmental_model is not None}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Main prediction pipeline: Fetch data, forecast AQI, predict price, 
    generate text advisory, and synthesize speech audio.
    """
    # 1. Fetch Live Data
    weather_data = get_live_weather(request.lat, request.lon)
    aqi_data = get_live_aqi(request.lat, request.lon)
    
    current_temp = weather_data.get("temperature_celsius") or 25.0
    current_humidity = weather_data.get("relative_humidity_percent") or 50.0
    current_precip = weather_data.get("precipitation_mm") or 0.0

    # 2. Model Inference Layer
    if environmental_model is None or price_model is None:
        raise HTTPException(status_code=500, detail="ML Models not loaded on the server.")

    # Predict future AQI using environmental model
    # Shape must match what the model expects, e.g., 2D array [[temp, humidity, precip]]
    env_inputs = np.array([[current_temp, current_humidity, current_precip]])
    forecasted_aqi = float(environmental_model.predict(env_inputs)[0])

    # Predict price using price model (mocked to predict distant market logic)
    # Keeping the original model input shape [[forecasted_aqi, current_price]]
    price_inputs = np.array([[forecasted_aqi, request.current_price]])
    predicted_price = float(price_model.predict(price_inputs)[0])
    predicted_price = max(request.current_price * 0.5, predicted_price)
    print(f"Prediction made using PKL model: {predicted_price}")
    
    # Advanced Profit Comparison
    local_revenue = request.current_price * request.yield_amount
    distant_profit = (request.distant_market_price * request.yield_amount) - request.transport_cost
    
    profit_improvement = distant_profit - local_revenue
    
    if profit_improvement > 0:
        recommended_action = "Transport to Distant Market"
    else:
        recommended_action = "Sell Locally"

    # 3. Explainability Layer (Featherless AI)
    if request.intent == "price_check":
        prompt = (
            f"Act as an agricultural expert. A farmer is growing {request.crop} at coordinates "
            f"({request.lat}, {request.lon}). "
            f"The local market price is {request.current_price}. "
            f"Generate an advisory focusing ONLY on the current market price. "
            f"You must write this entire advisory strictly in the {request.language} language."
        )
    elif request.intent == "climate_check":
        prompt = (
            f"Act as an agricultural expert. A farmer is growing {request.crop} at coordinates "
            f"({request.lat}, {request.lon}). "
            f"The current temperature is {current_temp}°C with an AQI of {aqi_data.get('aqi')}. "
            f"Generate an advisory focusing ONLY on the live weather data like Temperature and AQI. "
            f"You must write this entire advisory strictly in the {request.language} language."
        )
    else:
        prompt = (
            f"Act as an agricultural expert. A farmer is growing {request.crop} at coordinates "
            f"({request.lat}, {request.lon}) with an expected yield of {request.yield_amount}. "
            f"The local market price is {request.current_price}, yielding a local revenue of {local_revenue:.2f}. "
            f"The distant market price is {request.distant_market_price} with a transport cost of {request.transport_cost:.2f}, "
            f"yielding a distant profit of {distant_profit:.2f}. "
            f"The profit improvement if transported is {profit_improvement:.2f}. "
            f"The forecasted AQI is {forecasted_aqi:.2f}. "
            f"Based on this, the recommended action is: {recommended_action}. "
            f"Generate exactly 2 sentences of advice explicitly mentioning "
            f"the recommended action, transport cost, and profit improvement. "
            f"You must write this entire advisory strictly in the {request.language} language."
        )

    try:
        response = await featherless_client.chat.completions.create(
            model=FEATHERLESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        advisory_text = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Featherless AI API error: {str(e)}")

    # 4. Accessibility Layer (ElevenLabs)
    # Generate MP3 using eleven_multilingual_v2
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=advisory_text,
            voice_id="21m00Tcm4TlvDq8ikWAM", # Rachel voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Save to static folder
        filename = f"advisory_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join("static", filename)
        
        # Async Elevenlabs generate returns an async generator
        with open(filepath, "wb") as f:
            async for chunk in audio_stream:
                f.write(chunk)
                
        audio_url = f"/static/{filename}"
                
    except Exception as e:
        # If elevenlabs fails (e.g., missing API key), fallback gently
        print(f"ElevenLabs Audio Generation Error: {e}")
        audio_url = None

    # 5. Final Output
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
            "current_aqi": aqi_data.get("aqi")
        },
        "forecasts": {
            "forecasted_aqi": round(forecasted_aqi, 2),
            "predicted_price": round(predicted_price, 2),
            "profit_improvement": round(profit_improvement, 2),
            "recommended_action": recommended_action
        },
        "advisory": advisory_text,
        "audio_url": audio_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
