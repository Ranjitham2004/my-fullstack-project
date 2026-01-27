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
    print("Rainfall columns:", rainfall_df.columns.tolist())
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

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AGROMONITORING_APPID = os.getenv("AGROMONITORING_APPID")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome to CropWiseX"})

@app.get("/home", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/soil-details", response_class=HTMLResponse)
def soil_details_page(request: Request):
    soil_data = {
        "soil_temperature": 27.2,
        "soil_moisture": 0.1,
        "ph": 6.87,
        "n": 20.1,
        "p": 62.2,
        "k": 38.8,
        "soil_type": "Lo47-2a-3803"
    }
    return templates.TemplateResponse("soil_details.html", {"request": request, "soil": soil_data})

@app.get("/weather-details", response_class=HTMLResponse)
def weather_details_page(request: Request):
    weather_data = {
        "city": "Avadi",
        "temperature": 25.5,
        "humidity": 85,
        "description": "mist",
        "rainfall": 0.0
    }
    return templates.TemplateResponse("weather_details.html", {"request": request, "weather": weather_data})

@app.post("/save-location")
async def save_location(data: dict):
    lat = data.get("latitude")
    lon = data.get("longitude")
    
    if lat is None or lon is None:
        return JSONResponse({"status": "error", "message": "Invalid location data"}, status_code=400)

    try:
        lat = float(lat)
        lon = float(lon)
    except:
        return JSONResponse({"status": "error", "message": "Invalid lat/lon"}, status_code=400)

    print(f"Location: {lat}, {lon}")

    # Weather fetch
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        weather_data = response.json()
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        city = weather_data.get("name", "Your location")
        weather_info = {"city": city, "temperature": temp, "humidity": humidity, "description": description}
    except Exception as e:
        print("Weather API error:", str(e))
        return JSONResponse({"status": "error", "message": "Weather API error"}, status_code=500)

    # Rainfall estimation
    rainfall = weather_data.get('rain', {}).get('1h', 0.0)
    if rainfall == 0:
        desc_lower = description.lower()
        main_cond = weather_data["weather"][0]["main"].lower()
        if 'rain' in main_cond or 'drizzle' in desc_lower:
            if 'light' in desc_lower or 'drizzle' in desc_lower:
                rainfall = 1.0
            elif 'moderate' in desc_lower:
                rainfall = 5.0
            elif 'heavy' in desc_lower or 'shower' in desc_lower:
                rainfall = 15.0
            else:
                rainfall = 2.0
        elif 'thunderstorm' in desc_lower:
            rainfall = 20.0
    print(f"Rainfall before season: {rainfall} mm")

    # AgroMonitoring soil
    soil_info = {"ph": 7.0, "n": 200, "p": 40, "k": 160, "soil_temperature": 25.0, "soil_moisture": 30.0, "soil_type": "Unknown"}
    try:
        soil_url = f"https://api.agromonitoring.com/agro/1.0/soil?lat={lat}&lon={lon}&appid={AGROMONITORING_APPID}"
        soil_response = requests.get(soil_url)
        soil_response.raise_for_status()
        soil_data = soil_response.json()
        soil_moisture = soil_data.get('moisture', 30.0)
        soil_temp_k = soil_data.get('t0', 298.15)
        soil_temp = soil_temp_k - 273.15
        soil_info = {
            "ph": round(7.0, 2),
            "n": round(200, 1),
            "p": round(40, 1),
            "k": round(160, 1),
            "soil_temperature": round(soil_temp, 1),
            "soil_moisture": round(soil_moisture, 1),
            "soil_type": "Unknown"
        }
        print(f"AgroMonitoring: Temp {soil_info['soil_temperature']}°C, Moisture {soil_info['soil_moisture']}%")
    except Exception as e:
        print("AgroMonitoring error:", str(e))

    # HWSD soil code & name
    try:
        with rasterio.open("datasets/hwsd/hwsd.bil") as src:
            row, col = src.index(lon, lat)
            soil_code = src.read(1, window=((row, row+1), (col, col+1)))[0][0]
            print(f"HWSD code: {soil_code}")
        lookup_df = pd.read_csv("datasets/hwsd/GLOBAL_Soil.txt")
        soil_name = lookup_df[lookup_df['VALUE'] == soil_code]['NAME'].values
        soil_type = soil_name[0] if len(soil_name) > 0 else "Unknown"
        print(f"Soil type: {soil_type}")
        soil_info["soil_type"] = soil_type
    except Exception as e:
        print("HWSD error:", str(e))
        soil_info["soil_type"] = "Unknown"

    # District matching & nutrient override
    n, p, k, ph = 200, 40, 160, 7.0
    closest_district = None
    if soil_avg_df is not None:
        try:
            min_dist = float('inf')
            for dist_name, (d_lat, d_lon) in district_latlon.items():
                dist = ((d_lat - lat)**2 + (d_lon - lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_district = dist_name
            if closest_district and closest_district.lower() == "avadi":
                closest_district = "Chennai"
            district_row = soil_avg_df[soil_avg_df['District'].str.lower() == closest_district.lower()]
            if not district_row.empty:
                n = district_row['avg_n'].values[0]
                p = district_row['avg_p'].values[0]
                k = district_row['avg_k'].values[0]
                ph = district_row['avg_ph'].values[0]
                soil_info["n"] = round(n, 1)
                soil_info["p"] = round(p, 1)
                soil_info["k"] = round(k, 1)
                soil_info["ph"] = round(ph, 2)
                print(f"District {closest_district}: N {soil_info['n']}, P {soil_info['p']}, K {soil_info['k']}, pH {soil_info['ph']}")
        except Exception as e:
            print("District error:", str(e))

    # Season adjustment
    month = datetime.now().month
    if month in [10, 11, 12, 1, 2, 3]:
        rainfall *= 0.4
    elif month in [6, 7, 8, 9]:
        rainfall *= 2.0
    else:
        rainfall *= 0.8
    print(f"Final rainfall: {rainfall} mm")

    # NEW: Collect district insights from all uploaded files
    district_insights = {"district": closest_district or "Unknown"}

    # From rainfall_data.csv
    if rainfall_df is not None and closest_district:
        try:
            district_col = 'Unnamed: 1'
            print(f"Using rainfall district column: '{district_col}'")
            row = rainfall_df[rainfall_df[district_col].astype(str).str.strip().str.lower() == closest_district.lower()]
            if not row.empty:
                current_month_abbr = datetime.now().strftime("%b").upper()
                normal_col = next((col for col in row.columns if current_month_abbr in col and 'Normal' in col), None)
                actual_col = next((col for col in row.columns if current_month_abbr in col and 'Actual' in col), None)
                dev_col = next((col for col in row.columns if current_month_abbr in col and '% Dev' in col), None)

                if normal_col:
                    district_insights["normal_rainfall_mm"] = row[normal_col].values[0]
                if actual_col:
                    district_insights["actual_rainfall_mm"] = row[actual_col].values[0]
                if dev_col:
                    district_insights["rainfall_dev_percent"] = row[dev_col].values[0]
            else:
                print(f"No row found for district '{closest_district}' in rainfall data")
        except Exception as e:
            print("Rainfall lookup error:", str(e))
            print("Available columns:", rainfall_df.columns.tolist())

    # From land_use.csv — cultivable area
    if land_use_df is not None and closest_district:
        try:
            district_col = 'Unnamed: 1'
            print(f"Using land use district column: '{district_col}'")
            row = land_use_df[land_use_df[district_col].astype(str).str.strip().str.lower() == closest_district.lower()]
            if not row.empty:
                # Use actual column indices from your print (adjust if needed)
                district_insights["net_sown_area_ha"] = row.iloc[0, 2] if len(row.columns) > 2 else None
                district_insights["fallow_lands_ha"] = row.iloc[0, 9] if len(row.columns) > 9 else None
            else:
                print(f"No row found for district '{closest_district}' in land use data")
        except Exception as e:
            print("Land use lookup error:", str(e))
            print("Available columns:", land_use_df.columns.tolist())

    # From crop_production_history.csv — historical top crops
    historical_top = []
    if crop_history_df is not None:
        try:
            # Your columns are Unnamed: 0 to Unnamed: 11
            # Assume Unnamed: 1 is Crop, Unnamed: 2 to Unnamed: 11 are years
            crop_history_df['recent_avg'] = crop_history_df.iloc[:, 2:12].mean(axis=1, numeric_only=True)
            historical_top = crop_history_df.nlargest(5, 'recent_avg').iloc[:, 1].tolist()
        except Exception as e:
            print("Historical crops error:", str(e))
    district_insights["historical_top_crops"] = historical_top

    # Predict crop
    crop_prediction = None
    if model is not None:
        try:
            input_data = [[n, p, k, temp, humidity, ph, rainfall]]
            predicted_crop = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            top_prob = max(probabilities) * 100
            crop_prediction = {
                "recommended_crop": predicted_crop,
                "confidence": round(top_prob, 1)
            }
            print(f"Predicted: {predicted_crop} ({top_prob:.1f}%)")
        except Exception as e:
            print("Prediction error:", str(e))

    return JSONResponse({
        "status": "success",
        "message": f"Weather at {city}: {temp}°C, {humidity}% humidity, {description}",
        "weather": weather_info,
        "soil": soil_info,
        "rainfall": round(rainfall, 1),
        "district_insights": district_insights,
        "crop_prediction": crop_prediction or {"recommended_crop": "Unable to predict", "confidence": 0}
    })