# Inside your main.py prediction route...

# 1. Load the REAL master model at the top of main.py
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import numpy as np

try:
    master_price_model = joblib.load('master_price_model.pkl')
except Exception as e:
    print(f"Warning: Could not load master model. {e}")

# ... (inside your prediction endpoint) ...

# 2. Get Live Weather (You already do this!)
current_temp = live_climate.get("temperature_celsius", 30.0)
current_humidity = live_climate.get("relative_humidity_percent", 60.0)
current_precip = live_climate.get("precipitation_mm", 0.0)
current_aqi = aqi_data.get("aqi", 100)

# 3. Format the 14-Feature Array for the Master Model
# [price_lag1, price_lag3, price_roll3, price_volatility, aqi, rainfall, temp, humidity, crop_onion, crop_potato, crop_tomato, mandi_1, mandi_2, mandi_3]
base_historical_stats = [2100, 2050, 2080, 50] # Mocked historical data for demo

crop_name = request.crop.capitalize()
crop_arr = [
    1 if crop_name == "Onion" else 0,
    1 if crop_name == "Potato" else 0,
    1 if crop_name == "Tomato" else 0
]

# 4. Predict Prices across your 3 Mandis!
mandis = [
    {"name": "Coimbatore Mandi", "distance": 10, "encoded": [1, 0, 0]},
    {"name": "Erode Mandi", "distance": 105, "encoded": [0, 1, 0]},
    {"name": "Tiruppur Mandi", "distance": 55, "encoded": [0, 0, 1]}
]

best_profit = -999999
best_mandi = ""
best_price = 0

for mandi in mandis:
    # Build exact array expected by the model
    features = np.array([base_historical_stats + [current_aqi, current_precip, current_temp, current_humidity] + crop_arr + mandi["encoded"]])
    
    predicted_price = float(master_price_model.predict(features)[0])
    
    # Calculate Net Profit (Decision Layer)
    transport_cost = mandi["distance"] * 1.5 * 20 # 20 quintals
    gross_revenue = predicted_price * 20
    net_profit = gross_revenue - transport_cost
    
    if net_profit > best_profit:
        best_profit = net_profit
        best_mandi = mandi["name"]
        best_price = predicted_price

# 5. Send this back to your Frontend / ElevenLabs!
recommended_action = f"Based on live weather, sell at {best_mandi} for ₹{best_price:.2f} per quintal."
profit_improvement = best_profit - (2000 * 20) # Compare to a baseline
