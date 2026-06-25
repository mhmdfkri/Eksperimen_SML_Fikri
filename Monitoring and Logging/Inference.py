import requests

data = {
    "Monthly Charges": 80,
    "Tenure Months": 12
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)

print(response.json())