# scripts/predict_category.py

import pickle
import os
import re
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from scripts.text_cleaner import TextCleaner

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", str(text).lower())

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model uspešno učitan.\n")
except FileNotFoundError:
    print("Model nije pronađen. Pokreni train_model.py da ga kreiraš.")
    exit()
except Exception as e:
    print(f"Greška pri učitavanju modela: {e}")
    exit()

def predict(title):
    title_clean = clean_text(title)
    try:
        prediction = model.predict(pd.Series([title_clean]))[0]
        return prediction
    except Exception as e:
        return f"Greška pri predikciji: {e}"

if __name__ == "__main__":
    print("Interaktivni klasifikator proizvoda")
    print("Unesi naziv proizvoda (ili 'exit' za izlaz):\n")
    while True:
        user_input = input("📝 Naziv: ")
        if user_input.lower() == "exit":
            print("Izlaz iz programa.")
            break
        category = predict(user_input)
        print("Predviđena kategorija:", category, "\n")
