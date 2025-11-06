# ============================================================
# Smart Agriculture Monitoring System – Complete app.py
# ============================================================

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# ---------------- CONFIGURATION ----------------
UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH = os.path.join("model", "crop_disease_model.h5")
DB_PATH = os.path.join("database", "crops.db")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("model", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-this-with-a-secure-key"

INPUT_SIZE = 224

# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS crop_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    crop_name TEXT,
                    disease TEXT,
                    confidence REAL,
                    temperature REAL,
                    humidity REAL,
                    ph REAL,
                    moisture REAL,
                    lat REAL,
                    lon REAL
                )''')
    conn.commit()
    conn.close()

init_db()


def save_analysis_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO crop_history
                (timestamp, crop_name, disease, confidence, temperature, humidity, ph, moisture, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               data['crop_name'], data['disease'], data['confidence'],
               data['temperature'], data['humidity'], data['ph'],
               data['moisture'], data['lat'], data['lon']))
    conn.commit()
    conn.close()

# ---------------- MODEL LOADING ----------------
model = None
CLASS_NAMES = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        CLASS_NAMES = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
            'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        ]
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        model = None
else:
    print(f"⚠️ No model found at {MODEL_PATH}. Using fallback only.")


# ---------------- HELPER FUNCTIONS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(img_path, target_size=INPUT_SIZE):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((target_size, target_size))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_from_image(img_path):
    """
    Returns (label, confidence_percent, preds_array)
    confidence_percent: 0-100
    preds_array: array of probabilities if available
    """
    global model, CLASS_NAMES
    if not model:
        return None, None, None

    x = preprocess_image(img_path)
    preds = model.predict(x)
    if preds is None:
        return None, None, None

    # handle both (1, n) and (n,) shapes robustly
    arr = np.asarray(preds).ravel()
    idx = int(np.argmax(arr))
    conf = float(np.max(arr))
    label = (CLASS_NAMES[idx] if CLASS_NAMES and idx < len(CLASS_NAMES) else f"Class_{idx}")
    return label, conf * 100.0, arr  # convert to percentage


def get_live_weather(lat, lon):
    """
    Fetch real-time weather data using Open-Meteo (no API key required).
    Returns temperature (°C), humidity (%), and rainfall (mm, last hour or 0).
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current_weather=true&hourly=relativehumidity_2m,precipitation"
        )
        r = requests.get(url, timeout=8)
        data = r.json()

        current = data.get("current_weather", {})
        temp = current.get("temperature")
        # open-meteo returns hourly arrays in 'hourly' key; fallback to None if not present
        humidity = None
        rainfall = None
        hourly = data.get("hourly", {})
        if 'relativehumidity_2m' in hourly and 'time' in hourly:
            # try to find latest index matching current time
            rh_vals = hourly.get('relativehumidity_2m', [])
            if len(rh_vals) > 0:
                humidity = float(rh_vals[-1])
        if 'precipitation' in hourly:
            prec_vals = hourly.get('precipitation', [])
            if len(prec_vals) > 0:
                # sum last 24 hours if many points; here take latest hour
                rainfall = float(prec_vals[-1])
        return (round(temp, 1) if temp is not None else None,
                round(humidity, 1) if humidity is not None else None,
                round(rainfall, 2) if rainfall is not None else None)
    except Exception as e:
        print("⚠️ Weather fetch error:", e)
        return None, None, None


def get_soil_ph(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&depth=0-5cm"
        r = requests.get(url, timeout=8)
        data = r.json()
        layers = data.get("properties", {}).get("layers", [])
        if layers:
            vals = layers[0]["depths"][0]["values"]
            return round(vals.get("mean", 0) / 10.0, 2)
        return None
    except Exception as e:
        print("⚠️ Soil API error:", e)
        return None


def estimate_moisture(lat, lon):
    try:
        return float(40 + (int(abs(float(lat))) + int(abs(float(lon)))) % 30)
    except:
        return None


def generate_suggestion(temp, hum, ph, moisture, disease=None, rainfall=None):
    suggestions, fert_recommendations = [], []

    # --- pH suggestions ---
    if ph is not None:
        if ph < 5.5:
            suggestions.append("⚠️ Soil acidic — apply lime or compost.")
            fert_recommendations.append("Use NPK 20-20-20 or dolomite lime.")
        elif ph > 7.8:
            suggestions.append("⚠️ Soil alkaline — add organic matter.")
            fert_recommendations.append("Use ammonium sulfate or elemental sulfur.")
        else:
            suggestions.append("✅ Soil pH is balanced.")
            fert_recommendations.append("Maintain balanced NPK 10-26-26 every 20 days.")

    # --- Moisture suggestions ---
    if moisture is not None:
        if moisture < 30:
            suggestions.append("💧 Low moisture — irrigate soon or use drip system.")
        elif moisture > 80:
            suggestions.append("💦 Soil too wet — improve drainage to avoid fungal growth.")
        else:
            suggestions.append("💧 Moisture levels are optimal.")

    # --- Temperature & humidity combined logic ---
    if temp is not None and hum is not None:
        if temp > 35 and hum < 40:
            suggestions.append("🔥 Hot & dry — mulch soil and apply potassium nitrate spray.")
            fert_recommendations.append("Apply foliar potassium or humic acid to reduce stress.")
        elif temp < 20 and hum > 70:
            suggestions.append("🌫️ Cool & humid — high fungal risk, ensure ventilation.")
            fert_recommendations.append("Use Mancozeb or Copper oxychloride preventive spray.")
        elif hum > 85:
            suggestions.append("💦 High humidity — watch for leaf-spot diseases.")
        else:
            suggestions.append("🌤️ Weather conditions look favorable for growth.")

    # --- Rainfall logic ---
    if rainfall is not None:
        if rainfall > 50:
            suggestions.append("☔ Heavy rainfall detected — reduce irrigation & apply fungicide.")
        elif rainfall < 5:
            suggestions.append("🌞 Dry spell — schedule irrigation every 2–3 days.")

    # --- Disease-specific recommendations ---
    if disease:
        d = disease.lower()
        if "blight" in d:
            fert_recommendations.append("Spray Mancozeb or Copper fungicide (avoid watering leaves).")
        elif "rust" in d:
            fert_recommendations.append("Apply Propiconazole or Sulfur dusting under low humidity.")
        elif "mildew" in d:
            fert_recommendations.append("Use wettable sulfur or neem oil; avoid overhead watering.")
        elif "bacterial" in d:
            fert_recommendations.append("Use Streptomycin or Copper oxychloride; disinfect tools.")
        elif "virus" in d:
            fert_recommendations.append("Remove infected plants, control whiteflies, use Imidacloprid.")
        else:
            fert_recommendations.append("Maintain hygiene, monitor plants, rotate crops.")

    return suggestions or ["No major issues detected."], fert_recommendations or ["General care recommended."]


def get_disease_severity(confidence_percent, disease):
    """
    Convert prediction confidence (0-100) into a descriptive severity level.
    Higher confidence for a disease = more likely/detectable (so more severe).
    """
    if confidence_percent is None:
        return ("🔍 Unknown", "gray")
    if confidence_percent >= 90:
        return ("🔴 Severe (Immediate attention needed)", "red")
    elif 70 <= confidence_percent < 90:
        return ("🟡 Moderate (Needs monitoring)", "gold")
    else:
        return ("🟢 Mild / Early (Lower confidence)", "green")


def get_weather_insights(temp, humidity, rainfall, ph, moisture):
    insights = []
    condition_level = "Good"  # default

    if temp is not None:
        if temp > 35:
            insights.append("⚠️ High temperature may stress crops — ensure proper irrigation.")
            condition_level = "Moderate"
        elif temp < 15:
            insights.append("🌡️ Low temperature can slow plant growth — monitor for frost.")
            condition_level = "Moderate"

    if humidity is not None:
        if humidity > 80:
            insights.append("💧 High humidity detected — fungal diseases may spread faster.")
            condition_level = "Poor"
        elif humidity < 30:
            insights.append("Dry air — ensure sufficient irrigation and mulch to retain moisture.")
            condition_level = "Moderate"

    if rainfall is not None:
        if rainfall > 50:
            insights.append("🌧️ Heavy rainfall — check for waterlogging near roots.")
        elif rainfall < 5:
            insights.append("☀️ Little rainfall — consider additional watering or drip systems.")

    if ph is not None:
        if ph < 5.5:
            insights.append("Soil is acidic — add lime to balance pH.")
            condition_level = "Moderate"
        elif ph > 8:
            insights.append("Soil is alkaline — organic compost can help restore balance.")
            condition_level = "Moderate"

    if moisture is not None:
        if moisture < 25:
            insights.append("Low soil moisture — schedule irrigation soon.")
        elif moisture > 75:
            insights.append("Excess moisture — ensure proper drainage.")

    if not insights:
        insights.append("✅ All parameters within healthy range — good crop condition.")

    return insights, condition_level


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        crop_name = request.form.get("crop_name", "").strip()
        # Accept lat/lon from form; if not provided use safe fallback (Hyderabad)
        lat = request.form.get("lat") or 17.3850
        lon = request.form.get("lon") or 78.4867
        # Ensure numeric floats
        try:
            lat = float(lat)
            lon = float(lon)
        except:
            lat, lon = 17.3850, 78.4867

        file = request.files.get("crop_image")
        if not file or file.filename == "":
            flash("Please upload a crop image.")
            return redirect(url_for("home"))
        if not allowed_file(file.filename):
            flash("Unsupported file type.")
            return redirect(url_for("home"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # --- Live weather / soil / moisture ---
        temp, hum, rain = get_live_weather(lat, lon)
        ph = get_soil_ph(lat, lon)
        moisture = estimate_moisture(lat, lon)

        # --- Model Prediction ---
        disease_label, confidence_percent, preds = predict_from_image(save_path)
        # If model not found / prediction failed -> fallback
        if disease_label is None or confidence_percent is None:
            disease_label = "Model not available"
            confidence_percent = 50.0
            preds = np.array([1.0])  # dummy

        # --- Health Score (composite) ---
        # Use confidence_percent (0-100) as base, then add small influence from humidity, moisture and pH
        health_score = confidence_percent
        try:
            if hum is not None:
                health_score = health_score * 0.6 + hum * 0.1 + (moisture or 50) * 0.1 + (7 - abs((ph or 7) - 7)) * 2
                health_score = min(100, max(0, round(health_score, 2)))
        except:
            health_score = round(confidence_percent, 2)

        # --- Disease severity calculation ---
        severity_label, severity_color = get_disease_severity(confidence_percent, disease_label)

        # --- Simulate progression graph (7 points) ---
        x = np.arange(1, 8)
        # center around health_score but vary a little
        y = np.clip(health_score + np.random.randint(-8, 8, size=7), 0, 100)

        plt.figure(figsize=(4, 2.5))
        plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
        plt.title('Disease Progression Trend (Simulated)')
        plt.xlabel('Days')
        plt.ylabel('Health Score (%)')
        plt.grid(True)
        progression_graph_file = f"progression_{random.randint(1000,9999)}.png"
        progression_graph_path = os.path.join(app.config["UPLOAD_FOLDER"], progression_graph_file)
        plt.savefig(progression_graph_path, bbox_inches='tight')
        plt.close()

        # --- Confidence Distribution Graph ---
        if preds is None:
            preds = np.array([confidence_percent / 100.0])
        preds_arr = np.asarray(preds).ravel()
        labels = CLASS_NAMES if CLASS_NAMES else [f"Class_{i}" for i in range(len(preds_arr))]
        # If mismatch lengths, trim or expand labels
        if len(labels) != len(preds_arr):
            labels = [f"Class_{i}" for i in range(len(preds_arr))]

        plt.figure(figsize=(8, 3))
        plt.bar(range(len(preds_arr)), preds_arr * 100)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.title("Prediction Probabilities (%)")
        plt.tight_layout()
        graph_file = f"{filename}_graph.png"
        plt.savefig(os.path.join(app.config["UPLOAD_FOLDER"], graph_file))
        plt.close()

        # --- Suggestions ---
        general, fert_recommendations = generate_suggestion(temp, hum, ph, moisture, disease_label, rain)
        suggestions = {"general": general, "treatment": fert_recommendations}

        # --- Weather insights ---
        weather_insights, condition_level = get_weather_insights(temp, hum, rain, ph, moisture)

        # --- Save to DB (store health_score as confidence in DB) ---
        save_analysis_to_db({
            "crop_name": crop_name or disease_label,
            "disease": disease_label or "Unknown",
            "confidence": health_score,
            "temperature": temp,
            "humidity": hum,
            "ph": ph,
            "moisture": moisture,
            "lat": lat,
            "lon": lon
        })

        # --- Render to result.html ---
        return render_template(
            "result.html",
            crop_name=crop_name or disease_label,
            crop_note=None,
            model_loaded=(model is not None),
            prediction=disease_label or "Unknown",
            confidence=int(round(health_score)),
            temp=temp,
            humidity=hum,
            rainfall=rain,
            ph=ph,
            moisture=moisture,
            lat=lat,
            lon=lon,
            suggestions=suggestions,
            fert_recommendations=fert_recommendations,
            image_file=filename,
            graph_file=graph_file,
            severity_label=severity_label,
            severity_color=severity_color,
            progression_graph_file=progression_graph_file,
            weather_insights=weather_insights,
            condition_level=condition_level
        )

    except Exception as e:
        flash(f"⚠️ Error: {e}")
        return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM crop_history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("dashboard.html", records=rows)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


