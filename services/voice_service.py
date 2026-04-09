import os
from gtts import gTTS
import uuid

# This tells the code where the 'static/audio' folder is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")

# --- CRITICAL ADDITION: Ensure the folder exists ---
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR, exist_ok=True)
# --------------------------------------------------

def generate_voice(text, language_code):
    """
    Turns text into an MP3 file.
    language_code: 'ta' for Tamil, 'en' for English
    """
    try:
        # Create a unique name for the sound file
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Use Google to create the Tamil or English speech
        tts = gTTS(text=text, lang=language_code, slow=False)
        tts.save(filepath)

        # Return the web path so the browser can play it
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"Error generating voice: {e}")
        return None
