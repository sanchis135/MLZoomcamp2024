import requests

patient = {
    'age': 37,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 187,
    'exang': 0,
    'oldpeak': 3.5,
    'slope': 0,
    'ca': 0,
    'thal': 2 
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient).json()
print(response)

if response['target'] == True:
    print('The patient is illness of the heart.')
else:
    print('The patient is not illness of the heart.')