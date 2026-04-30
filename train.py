import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("data/housing.csv")

# -----------------------------
# 2️⃣ Clean column names (SAFE FIX)
# -----------------------------
data.columns = data.columns.str.strip().str.lower()

# Ensure correct column name
# (fixes "loacation" typo issue if present)
if "loacation" in data.columns:
    data.rename(columns={"loacation": "location"}, inplace=True)

# -----------------------------
# 3️⃣ Encode location
# -----------------------------
le = LabelEncoder()
data["location"] = le.fit_transform(data["location"])

# -----------------------------
# 4️⃣ Features & target
# -----------------------------
X = data[["location", "area", "bedrooms", "bathrooms"]]
y = data["price"]

# -----------------------------
# 5️⃣ Train model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 6️⃣ Save model + encoder
# -----------------------------
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/location_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained successfully with LOCATION + saved!")