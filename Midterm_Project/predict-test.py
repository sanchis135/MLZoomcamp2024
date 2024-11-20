import requests
import json

customer = {
    'Call Failure': 20,
    'Complains': 5,
    'Subscription Length': 15,
    'Charge Amount': 1,
    'Frequency of use': 10,
    'Frequency of SMS': 10,
    'Distinct Called Numbers': 5,
    'Tariff Plan': 1,
    'Status': 1,
    'Age': 30,
    'Customer Value': 50
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=customer)
result = response.json()

print(json.dumps(result, indent=2))
