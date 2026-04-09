from fastapi import APIRouter
from fastapi.responses import JSONResponse
import requests
import pandas as pd
from datetime import datetime
import os
import rasterio
import google.generativeai as genai

from main import (
    soil_avg_df, district_latlon, model,
    tn_crop_prod_df, agri_yield_df, rainfall_df,
    land_use_df, crop_history_df
)

import main   

router = APIRouter()


# ==================== AI FARMING SCHEDULE (used for both GPS and Manual) ====================
async def generate_ai_farming_schedule(crop_name: str, soil_info: dict, weather: dict, rainfall: float, district: str):
    prompt = f"""
You are a senior Tamil Nadu Agriculture Officer from TNAU.
Give practical farming advice for **{crop_name}** in **{district}, Tamil Nadu**.

Current conditions:
- Temperature: {weather.get('temperature', 28)}°C
- Humidity: {weather.get('humidity', 70)}%
- Rainfall: {rainfall} mm
- Soil Moisture: {soil_info.get('soil_moisture', 30)}%
- Soil pH: {soil_info.get('ph', 7.0)}
- Soil Type: {soil_info.get('soil_type', 'Loam')}

Return **only** in this exact format (no extra text):

Fertilizer Schedule: [your answer]
Irrigation Advice: [your answer]
Harvest Time: [your answer]
Soil Recovery / Rotation: [your answer]
Estimated Profit: [₹ amount per hectare]
"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()

        schedule = {
            "fertilizer_advice": "N/A",
            "irrigation_advice": "N/A",
            "harvest_advice": "N/A",
            "recovery_advice": "N/A",
            "profit_advice": "N/A"
        }

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("Fertilizer Schedule:"):
                schedule["fertilizer_advice"] = line.split(":", 1)[1].strip()
            elif line.startswith("Irrigation Advice:"):
                schedule["irrigation_advice"] = line.split(":", 1)[1].strip()
            elif line.startswith("Harvest Time:"):
                schedule["harvest_advice"] = line.split(":", 1)[1].strip()
            elif "Recovery" in line or "Rotation" in line:
                schedule["recovery_advice"] = line.split(":", 1)[1].strip()
            elif line.startswith("Estimated Profit:"):
                schedule["profit_advice"] = line.split(":", 1)[1].strip()

        return schedule

    except Exception as e:
        print("AI Schedule Error:", str(e))
        return {
            "fertilizer_advice": f"For {crop_name}: Use general NPK 80:40:60 kg/ha.",
            "irrigation_advice": f"For {crop_name}: Water every 5-7 days.",
            "harvest_advice": f"For {crop_name}: Typical harvest in 60–120 days.",
            "recovery_advice": f"After {crop_name}: Rotate with legumes.",
            "profit_advice": f"Estimated profit for {crop_name}: Check local market."
        }


# ==================== MAIN PROCESS FUNCTION ====================
async def process_location(lat: float, lon: float, city_name: str = None, use_sensor: bool = False):
    print(f"Processing location: {lat}, {lon} | Manual City: {city_name}")

    # Weather
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        weather_data = response.json()
        # ✅ Use sensor ONLY for GPS
        print("Sensor values:", main.latest_temperature, main.latest_humidity)
        if use_sensor and main.latest_temperature is not None and main.latest_humidity is not None:
            temp = main.latest_temperature
            humidity = main.latest_humidity
            print("✅ Using SENSOR data")
        else:
            temp = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            print("🌍 Using API data")
        description = weather_data["weather"][0]["description"]
        city = city_name or weather_data.get("name", "Your location")
        weather_info = {"city": city, "temperature": temp, "humidity": humidity, "description": description}
    except:
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

    # Soil Info
    soil_info = {"ph": 7.0, "n": 200, "p": 40, "k": 160, "soil_temperature": 25.0, "soil_moisture": 30.0, "soil_type": "Unknown"}
    try:
        soil_url = f"https://api.agromonitoring.com/agro/1.0/soil?lat={lat}&lon={lon}&appid={os.getenv('AGROMONITORING_APPID')}"
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
    except Exception as e:
        print("AgroMonitoring error:", str(e))

    # HWSD soil
    try:
        with rasterio.open("datasets/hwsd/hwsd.bil") as src:
            row, col = src.index(lon, lat)
            soil_code = src.read(1, window=((row, row+1), (col, col+1)))[0][0]
        lookup_df = pd.read_csv("datasets/hwsd/GLOBAL_Soil.txt")
        soil_name = lookup_df[lookup_df['VALUE'] == soil_code]['NAME'].values
        soil_type = soil_name[0] if len(soil_name) > 0 else "Unknown"
        soil_info["soil_type"] = soil_type
    except Exception as e:
        print("HWSD error:", str(e))
        soil_info["soil_type"] = "Unknown"

    # District matching & nutrient override (your original code)
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
        except Exception as e:
            print("District error:", str(e))

    # Show exact address you typed in manual box
    if city_name:
        print(f"✅ Using manual address: {city_name}")
    else:
        print(f"✅ Using district: {closest_district}")

    # Season adjustment
    month = datetime.now().month
    if month in [10, 11, 12, 1, 2, 3]:
        rainfall *= 0.4
    elif month in [6, 7, 8, 9]:
        rainfall *= 2.0
    else:
        rainfall *= 0.8
    print(f"Final rainfall: {rainfall} mm")

    # District insights from all your datasets
    district_insights = {"district": closest_district or "Unknown"}

    # Rainfall data
    if rainfall_df is not None and closest_district:
        try:
            district_col = 'Unnamed: 1'
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
        except Exception as e:
            print("Rainfall lookup error:", str(e))

    # Top crops from your datasets
    top_crops = []
    if tn_crop_prod_df is not None and closest_district:
        df = tn_crop_prod_df[tn_crop_prod_df['District'].str.strip().str.lower() == closest_district.lower()]
        if not df.empty:
            top_crops = df.groupby('Crop')['Area'].sum().nlargest(5).index.tolist()
    if agri_yield_df is not None and closest_district:
        df = agri_yield_df[agri_yield_df['District_Name'].str.strip().str.lower() == closest_district.lower()]
        if not df.empty:
            extra_top = df.groupby('Crop')['Area'].sum().nlargest(5).index.tolist()
            top_crops = list(set(top_crops + extra_top))[:5]
    district_insights["top_crops"] = top_crops

    # Land use
    if land_use_df is not None and closest_district:
        try:
            district_col = 'Unnamed: 1'
            row = land_use_df[land_use_df[district_col].astype(str).str.strip().str.lower() == closest_district.lower()]
            if not row.empty:
                district_insights["net_sown_area_ha"] = row.get('Net area sown', pd.Series([None])).values[0]
                district_insights["fallow_lands_ha"] = row.get('Fallow lands other than current fallow', pd.Series([None])).values[0]
        except Exception as e:
            print("Land use lookup error:", str(e))

    # Historical top crops
    historical_top = []
    if crop_history_df is not None:
        recent_cols = ['2017-18', '2018-19', '2019-20']
        if all(col in crop_history_df.columns for col in recent_cols):
            crop_history_df['recent_avg'] = crop_history_df[recent_cols].mean(axis=1, numeric_only=True)
            historical_top = crop_history_df.nlargest(5, 'recent_avg')['Crop'].tolist()
    district_insights["historical_top_crops"] = historical_top

    # ==================== CROP RECOMMENDATION (Dataset logic for both GPS and Manual) ====================
    crop_prediction = None
    if model is not None:
        try:
            input_df = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            probabilities = model.predict_proba(input_df)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            all_crops = model.classes_

            top_crops_list = []
            for idx in top_indices:
                crop_name = all_crops[idx]
                conf = round(probabilities[idx] * 100, 1)
                top_crops_list.append({"crop": crop_name.capitalize(), "confidence": conf})

            # Priority to district top crops from your datasets
            # FIXED: Replaced all general names with specific major crops of Tamil Nadu
            district_top = district_insights.get("top_crops", []) + district_insights.get("historical_top_crops", [])
            if district_top:
                for d_crop in district_top[:5]:   # check more to find good ones
                    crop_name = str(d_crop).strip()

                    # Replace general names with specific crops
                    if crop_name.lower() in ["total foodgrain", "foodgrain", "cereals"]:
                        crop_name = "Rice"
                    elif crop_name.lower() in ["other oilseeds", "oilseeds", "other oilseed"]:
                        crop_name = "Groundnut"
                    elif crop_name.lower() in ["pulses", "other pulses"]:
                        crop_name = "Moong (green gram)"
                    elif crop_name.lower() in ["other non food crops", "non food crops"]:
                        crop_name = "Coconut"

                    top_crops_list.insert(0, {"crop": crop_name.capitalize(), "confidence": 82})
                    if len(top_crops_list) >= 3:
                        break

            crop_prediction = {
                "recommended_crops": top_crops_list[:3],
                "recommended_crop": top_crops_list[0]["crop"]
            }
        except Exception as e:
            print("Prediction error:", str(e))

    # ==================== AI FARMING SCHEDULE ====================
    if crop_prediction and crop_prediction.get("recommended_crops"):
        primary_crop = crop_prediction["recommended_crops"][0]["crop"]
        schedule = await generate_ai_farming_schedule(
            crop_name=primary_crop,
            soil_info=soil_info,
            weather=weather_info,
            rainfall=rainfall,
            district=closest_district
        )

        district_insights["fertilizer_advice"] = schedule["fertilizer_advice"]
        district_insights["irrigation_advice"] = schedule["irrigation_advice"]
        district_insights["harvest_advice"] = schedule["harvest_advice"]
        district_insights["recovery_advice"] = schedule["recovery_advice"]
        district_insights["profit_advice"] = schedule["profit_advice"]

    return JSONResponse({
        "status": "success",
        "weather": weather_info,
        "soil": soil_info,
        "rainfall": round(rainfall, 1),
        "district_insights": district_insights,
        "crop_prediction": crop_prediction or {"recommended_crops": []},

        # 🔥 ADD THIS
        "sensor_data": {
            "temperature": main.latest_temperature,
            "humidity": main.latest_humidity
        }
    })


@router.post("/save-location")
async def save_location(data: dict):
    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        return JSONResponse({"status": "error", "message": "Invalid location"}, status_code=400)
    return await process_location(float(lat), float(lon), use_sensor=True)


@router.post("/save-manual-address")
async def save_manual_address(data: dict):
    address = data.get("address", "").strip()
    if not address:
        return JSONResponse({"status": "error", "message": "Address required"}, status_code=400)

    geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={address}&limit=1&appid={os.getenv('OPENWEATHER_API_KEY')}"
    try:
        geo_data = requests.get(geo_url, timeout=8).json()
        if not geo_data:
            return JSONResponse({"status": "error", "message": "Location not found"}, status_code=400)
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city = geo_data[0].get("name", address)
    except:
        return JSONResponse({"status": "error", "message": "Geocoding failed"}, status_code=500)

    return await process_location(lat, lon, city)
