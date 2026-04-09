# 🌱 CropWiseX

## 🚀 Features
- Crop Recommendation System
- AI Chatbot
- Weather Integration (OpenWeather)
- Voice Interaction
- Data-driven insights

## 🛠️ Tech Stack
- Backend: FastAPI (Python)
- Frontend: Jinja2 templates and Bootstrap
- Machine Learning: Scikit-learn / Pandas
- APIs: OpenWeather, AgroMonitoring, OpenRouter

## ⚙️ Local Setup

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📦 Environment Variables

Create a local `.env` file with:

```env
OPENWEATHER_API_KEY=your_openweather_key
AGROMONITORING_APPID=your_agromonitoring_key
OPENROUTER_API_KEY=your_openrouter_key
```

## 🚀 Deploy to Render

This repo already includes `render.yaml` for Render deployment.

1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Choose the `main` branch.
4. Render will use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add the same environment variables in Render's dashboard.
