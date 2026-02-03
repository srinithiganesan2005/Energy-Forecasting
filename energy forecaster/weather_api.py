import requests

API_KEY = "0f70b77bc3b6ed8477307d802a52bcf1"
CITY = "Coimbatore"

def get_temperature():
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={CITY}&appid={API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        return round(data["main"]["temp"], 2)
    except:
        return 30.0  # fallback temperature
