import os
import joblib

print("Current directory:")
print(os.getcwd())

print("\nIsi folder saat ini:")
print(os.listdir())

model = joblib.load(
    r"C:\Users\ACER\Desktop\Eksperimen_SML_Fikri\training\random_forest.pkl"
)

print("\nModel berhasil dibuka")
print(type(model))