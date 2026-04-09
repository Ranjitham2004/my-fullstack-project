from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import rasterio
from fastapi import UploadFile, File, HTTPException, status
from loguru import logger
import httpx
from typing import Tuple, List
from fastapi import APIRouter, Request, Body, HTTPException
from pydantic import BaseModel
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import uuid
from openai import OpenAI
from services.voice_service import generate_voice
import google.generativeai as genai

# Latest sensor readings from ESP32 (updated by /save-sensor-data)
latest_temperature = None
latest_humidity = None
latest_moisture = None

load_dotenv()
app = FastAPI()

# Load soil dataset
try:
    soil_df = pd.read_csv("datasets/Soil data.csv")
    print("Soil dataset loaded successfully! Rows:", len(soil_df))
    print("Columns in dataset:", soil_df.columns.tolist())
except Exception as e:
    print("Error loading soil dataset:", str(e))
    soil_df = None

# Calculate district averages
soil_avg_df = None
if soil_df is not None:
    soil_avg_df = soil_df.groupby('District').agg({
        'Nitrogen Value': 'mean',
        'Phosphorous value': 'mean',
        'Potassium value': 'mean',
        'pH': 'mean'
    }).reset_index()
    soil_avg_df.columns = ['District', 'avg_n', 'avg_p', 'avg_k', 'avg_ph']
    print("District soil averages calculated! Number of districts:", len(soil_avg_df))

# Load all new uploaded datasets
tn_crop_prod_df = None
rice_prod_df = None
crop_history_df = None
rainfall_df = None
land_use_df = None
agri_yield_df = None

try:
    tn_crop_prod_df = pd.read_csv("datasets/Tamilnadu Crop-Production.csv")
    print("Tamilnadu Crop-Production loaded! Rows:", len(tn_crop_prod_df))
except Exception as e:
    print("Error loading Tamilnadu Crop-Production:", str(e))

try:
    rice_prod_df = pd.read_csv("datasets/rice_production.csv")
    print("rice_production loaded! Rows:", len(rice_prod_df))
except Exception as e:
    print("Error loading rice_production:", str(e))

try:
    crop_history_df = pd.read_csv("datasets/crop_production_history.csv")
    print("crop_production_history loaded! Rows:", len(crop_history_df))
    print("Crop history columns:", crop_history_df.columns.tolist())
except Exception as e:
    print("Error loading crop_production_history:", str(e))

try:
    rainfall_df = pd.read_csv("datasets/rainfall_data.csv")
    print("rainfall_data loaded! Rows:", len(rainfall_df))
except Exception as e:
    print("Error loading rainfall_data:", str(e))

try:
    land_use_df = pd.read_csv("datasets/land_use.csv")
    print("land_use loaded! Rows:", len(land_use_df))
    print("Land use columns:", land_use_df.columns.tolist())
except Exception as e:
    print("Error loading land_use:", str(e))

try:
    agri_yield_df = pd.read_csv("datasets/Tamilnadu agriculture yield data.csv")
    print("Tamilnadu agriculture yield data loaded! Rows:", len(agri_yield_df))
except Exception as e:
    print("Error loading Tamilnadu agriculture yield data:", str(e))

# District lat/long dictionary (exact names from your CSV)
district_latlon = {
    "Ariyalur": (11.1385, 79.0779),
    "Chengalpattu": (12.6833, 79.9833),
    "Coimbatore": (11.0168, 76.9558),
    "Dindigul": (10.3687, 77.9803),
    "Erode": (11.3410, 77.7172),
    "Kanchipuram": (12.8352, 79.6993),
    "Kanniyakumari": (8.0883, 77.5385),
    "Karur": (10.9596, 78.0766),
    "Madurai": (9.9252, 78.1198),
    "Nagapattinam": (10.7662, 79.8449),
    "Namakkal": (11.2195, 78.1676),
    "Perambalur": (11.2333, 78.8667),
    "Pudukkottai": (10.4500, 78.8167),
    "Ramanathapuram": (9.3713, 78.8314),
    "Salem": (11.6643, 78.1460),
    "Sivaganga": (9.8433, 78.4803),
    "Thanjavur": (10.786999, 79.137827),
    "The Nilgiris": (11.4917, 76.7333),
    "Theni": (10.0104, 77.4770),
    "Thiruvallur": (13.1433, 79.9083),
    "Thiruvarur": (10.7672, 79.6350),
    "Tiruchippalli": (10.7905, 78.7047),
    "Tirunelveli": (8.7139, 77.7567),
    "Tirupathur": (12.4935, 78.2132),
    "Tiruppur": (11.1085, 77.3411),
    "Tiruvannamalai": (12.2250, 79.0747),
    "Tuticorn": (8.7642, 78.1348),
    "Vellore": (12.9165, 79.1325),
    "Villupuram": (11.9395, 79.4924),
    "Virudhunagar": (9.5790, 77.9584),
}

# Load crop recommendation dataset and train model
try:
    crop_df = pd.read_csv("datasets/Crop_recommendation.csv")
except Exception as e:
    print("Error loading Crop_recommendation:", str(e))
    crop_df = None

model = None
if crop_df is not None:
    features = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    target = crop_df['label']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Crop ML model trained! Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AGROMONITORING_APPID = os.getenv("AGROMONITORING_APPID")

# =========================
# FRONTEND PAGE ROUTES
# =========================

@app.get("/", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", context={"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/disease", response_class=HTMLResponse)
def disease_page(request: Request):
    return templates.TemplateResponse("disease.html", context={"request": request})

# ────────────────────────────────────────────────
# NEW FEATURE: Plant Disease Detection (from Plant-AI)
# Uses HuggingFace Inference API - no local model download needed
# ────────────────────────────────────────────────

@app.post("/api/detect-disease")
async def detect_disease(image: UploadFile = File(...)):
    try:
        contents = await image.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty image")

        # Convert image to base64
        import base64
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # OpenRouter Vision Model
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        prompt = """
You are an expert agricultural scientist.

Analyze the uploaded plant leaf image and respond strictly in this format:

Crop: <crop name>
Disease: <disease name or Healthy>
Confidence: <percentage>
Cause: <short reason>
Treatment:
- <point 1>
- <point 2>
- <point 3>

Important:
- Be accurate and practical
- If unsure, say "Possible <disease>"
- Give specific treatment (fertilizer / fungicide / organic solution)
"""

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        result_text = response.choices[0].message.content

        # 🔥 Parse response
        lines = result_text.split("\n")

        crop = "Unknown"
        disease = "Unknown"
        confidence = "N/A"
        recommendations = []

        for line in lines:
            if "Crop:" in line:
                crop = line.split("Crop:")[-1].strip()
                crop = crop.replace("Moringa", "Drumstick")
            elif "Disease:" in line:
                disease = line.split("Disease:")[-1].strip()
                disease = disease.replace("Possible", "").strip()
            elif "Confidence:" in line:
                confidence = line.split("Confidence:")[-1].strip()
            elif "-" in line:
                recommendations.append(line.replace("-", "").strip())

        return {
            "status": "success",
            "disease": f"{crop} - {disease}",
            "confidence": confidence,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ────────────────────────────────────────────────
# NEW ROUTE: Separate Disease Detection Page
# Uses diseasenew.html to avoid conflict with existing /disease
# ────────────────────────────────────────────────

@app.get("/diseasenew")
async def disease_new_page(request: Request):
    return templates.TemplateResponse("diseasenew.html", context={"request": request})

# Register crop router (this was missing!)
from routers.crop_router import router
app.include_router(router)

# CHATBOT

# 1. Define the Request Model ONCE (Cleaned up duplicates)
class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    language: str = "English"

# 2. Page Route to load the Chat Interface
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", context={"request": request})

# 3. Main Chat API Endpoint
@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    try:
        # Import the logic only when needed to avoid circular imports
        from services.chatbot_service import chat_with_memory
        from services.voice_service import generate_voice

        # A. Handle Session ID
        session_id = request.session_id or str(uuid.uuid4())

        # B. Get Text Response (Your existing logic)
        reply = await chat_with_memory(session_id, request.message)

        # C. --- VOICE OVER FEATURE START ---
        # Pick 'ta' if Tamil characters are present, else 'en'
        is_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in reply)
        lang_code = 'ta' if is_tamil else 'en'
        
        # Link to voice_service.py to create the MP3
        audio_url = generate_voice(reply, lang_code)
        # --- VOICE OVER FEATURE END ---

        # D. Return the keys your chat.html expects
        return {
            "reply": reply,
            "audio_url": audio_url,
            "session_id": session_id
        }

    except Exception as e:
        print(f"Chat Error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"reply": "I'm having trouble connecting to the AI.", "error": str(e)}
        )


# ================= SENSOR DATA API =================
from fastapi import Body

@app.post("/save-sensor-data")
async def save_sensor_data(data: dict = Body(...)):
    global latest_temperature, latest_humidity

    latest_temperature = data.get("temperature")
    latest_humidity = data.get("humidity")

    print("Received Sensor Data:", latest_temperature, latest_humidity)

    return {"status": "success"}

@app.get("/get-sensor-data")
async def get_sensor_data():
    return {
        "temperature": latest_temperature,
        "humidity": latest_humidity
    }

@app.get("/hardware", response_class=HTMLResponse)
async def hardware_page(request: Request):
    return templates.TemplateResponse("hardware.html", context={"request": request})


@app.post("/hardware-predict")
async def hardware_predict(data: dict):

    if model is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame([[ 
        data["N"], data["P"], data["K"],
        data["temperature"], data["humidity"],
        data["ph"], data["rainfall"]
    ]], columns=['N','P','K','temperature','humidity','ph','rainfall'])

    # 🔥 Get probabilities
    probabilities = model.predict_proba(input_df)[0]
    crops = model.classes_

    # 🔥 Get top 3 crops
    top_indices = probabilities.argsort()[-3:][::-1]

    top_crops = []
    for idx in top_indices:
        top_crops.append(crops[idx])

    return {
        "top_crops": top_crops
    }
