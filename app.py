from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import pickle
import os

# ---------------------------------
# APP CONFIGURATION
# ---------------------------------
app = Flask(__name__)
app.secret_key = "secret123"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------------------------
# DATABASE MODELS
# ---------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    area = db.Column(db.Float, nullable=False)
    bedrooms = db.Column(db.Float, nullable=False)
    bathrooms = db.Column(db.Float, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print("Error: model.pkl not found!")
        return None

model = load_model()

# ---------------------------------
# LOCATION MAP
# ---------------------------------
location_map = {
    "guntur": 0,
    "vijayawada": 1,
    "vizag": 2,
    "tenali": 3,
    "amaravathi": 4,
    "lam": 5
}

# ---------------------------------
# LOGIN ROUTE
# ---------------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user"] = user.username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid Username or Password")

    return render_template("login.html")

# ---------------------------------
# REGISTER ROUTE
# ---------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            return render_template("register.html", error="Username already exists!")

        if User.query.filter_by(email=email).first():
            return render_template("register.html", error="Email already exists!")

        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

# ---------------------------------
# HOME PAGE
# ---------------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("index.html", username=session["user"])

# ---------------------------------
# PREDICT ROUTE
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        location = request.form.get("location").lower()
        area = float(request.form.get("area"))
        bedrooms = float(request.form.get("bedrooms"))
        bathrooms = float(request.form.get("bathrooms"))

        location_encoded = location_map.get(location, 0)

        features = np.array([[location_encoded, area, bedrooms, bathrooms]])
        prediction = float(model.predict(features)[0])

        record = PredictionHistory(
            username=session["user"],
            location=location,
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            predicted_price=prediction
        )

        db.session.add(record)
        db.session.commit()

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ₹ {round(prediction, 2)}",
            username=session["user"]
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            username=session["user"]
        )

# ---------------------------------
# PROFILE PAGE
# ---------------------------------
@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]
    user = User.query.filter_by(username=username).first()

    total_predictions = PredictionHistory.query.filter_by(username=username).count()

    return render_template(
        "profile.html",
        user=user,
        total_predictions=total_predictions
    )

# ---------------------------------
# HISTORY PAGE
# ---------------------------------
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    records = PredictionHistory.query.filter_by(username=username)\
        .order_by(PredictionHistory.timestamp.desc()).all()

    return render_template("history.html", records=records)

# ---------------------------------
# DELETE HISTORY
# ---------------------------------
@app.route("/delete_history", methods=["POST"])
def delete_history():
    if "user" not in session:
        return redirect(url_for("login"))

    username = session["user"]

    PredictionHistory.query.filter_by(username=username).delete()
    db.session.commit()

    return redirect(url_for("history"))

# ---------------------------------
# LOGOUT
# ---------------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)