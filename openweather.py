import json
import requests
import datetime



def get_api_key():
    with open('api_key.json') as f: return json.load(f)["key"]


def save_api_key(api_key):
    with open('api_key.json', 'w') as f:
        json.dump({"key": api_key}, f)


class OpenWeather:
    API_URL = "http://api.openweathermap.org/data/2.5/"

    def __init__(self, api_key=None):
        if api_key is None: api_key = get_api_key()
        else: save_api_key(api_key)
        self.api_key = api_key


    def forecast(self, lat, lon):
        url = f"{self.API_URL}forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"

        response = requests.get(url).json()
        if response["cod"] != '200':
            print(response)
            raise Exception("Failed to fetch data")

        forecast = []
        for prevision in response['list']:
            epoch = prevision['dt']
            date = datetime.datetime.fromtimestamp(epoch).strftime('%d/%m/%y %H:%M:%S')
            temp = prevision['main']['temp']
            humidity = prevision['main']['humidity']

            vento_dir = prevision['wind']['deg']
            vento_vel = prevision['wind']['speed']
            vento_raj = prevision['wind']['gust']

            precipitation = prevision['rain']['3h'] if 'rain' in prevision else 0.0
            pressure = prevision['main']['grnd_level']

            forecast.append([date, temp, humidity, vento_dir, vento_vel, vento_raj, precipitation, pressure])

        return forecast
    

if __name__ == "__main__":
    openweather = OpenWeather()
    forecast = openweather.forecast(-16.6869, -49.2648)

    for prevision in forecast:
        print(prevision)