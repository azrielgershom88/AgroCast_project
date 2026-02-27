import requests

def get_live_weather(lat: float, lon: float) -> dict:
    """
    Fetches current weather data (temperature, humidity, precipitation) 
    for a given latitude and longitude using the Open-Meteo API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        
        return {
            "temperature_celsius": current.get("temperature_2m"),
            "relative_humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return {
            "temperature_celsius": None,
            "relative_humidity_percent": None,
            "precipitation_mm": None,
        }

def get_live_aqi(lat: float, lon: float) -> dict:
    """
    Fetches current Air Quality Index (AQI) data for a given 
    latitude and longitude using the Open-Meteo Air Quality API.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["european_aqi"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        
        return {
            "aqi": current.get("european_aqi")
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching AQI data: {e}")
        return {
            "aqi": None
        }

if __name__ == "__main__":
    # Example usage for New Delhi
    lat, lon = 28.6139, 77.2090
    print("Weather:", get_live_weather(lat, lon))
    print("AQI:", get_live_aqi(lat, lon))
