import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

chat_sessions = {}

# 🔹 Clean unwanted symbols like ** ###
def clean_text(text):
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"#+", "", text)
    return text.strip()

async def chat_with_memory(session_id, message):

    # Detect Tamil
    is_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in message)

    # ✅ SYSTEM PROMPTS
    if is_tamil:
        system_prompt = """
நீங்கள் 'CropWiseX' - ஒரு அனுபவம் வாய்ந்த விவசாய நிபுணர்.

விவசாயிகள் எளிதில் புரிந்து கொள்ளக்கூடிய மிகவும் தெளிவான தமிழில் பதில் அளிக்கவும்.

விதிமுறைகள்:
1. முழுக்க தமிழில் எழுதவும் (அவசியமான இடங்களில் மட்டும் ஆங்கில பெயர் சேர்க்கலாம்)
2. கலந்த மொழி (Tanglish) பயன்படுத்த வேண்டாம்
3. எளிய, இயல்பான, பேசும் தமிழ் பயன்படுத்தவும்
4. ஒவ்வொரு கேள்விக்கும் விரிவாக பதில் அளிக்கவும் (மிக குறுகிய பதில் வேண்டாம்)

5. நோய்கள் பற்றிய கேள்வி என்றால்:
   - நோய் பெயர்
   - அறிகுறி
   - காரணம்
   - தீர்வு (படிநிலையாக)
   - பயன்படுத்த வேண்டிய மருந்துகள்

6. உரங்கள்/மருந்துகள் பற்றிய கேள்வி என்றால்:
   - பயன்பாடு
   - எப்போது பயன்படுத்த வேண்டும்
   - எப்படி பயன்படுத்த வேண்டும்

7. பட்டியல் வடிவில் (1, 2, 3...) தெளிவாக எழுதவும்
8. Markdown symbols (**, ##) பயன்படுத்த வேண்டாம்
9. விவசாயிக்கு நடைமுறையில் உதவும் தகவல்கள் மட்டும் கொடுக்கவும்

பதில் தெளிவாகவும், சற்று விரிவாகவும் இருக்க வேண்டும்.
"""
    else:
        system_prompt = """
You are 'CropWiseX', an expert Agriculture Advisor.

Give clear, detailed, and farmer-friendly answers.

Guidelines:
1. Use simple and easy-to-understand English
2. Do NOT give very short answers — explain properly
3. Avoid symbols like ** or ##

4. For disease-related questions, ALWAYS include:
   - Disease name
   - Symptoms
   - Causes
   - Step-by-step solutions
   - Recommended pesticides/fungicides

5. For fertilizer-related questions:
   - Purpose of the fertilizer
   - When to use
   - How to apply

6. Use numbered steps (1, 2, 3...)
7. Focus on practical farming advice
8. Keep answers structured and slightly detailed

The answer should be clear enough for farmers to understand easily.
"""

    # Initialize session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [
            {"role": "system", "content": system_prompt}
        ]

    # Add user message
    chat_sessions[session_id].append({
        "role": "user",
        "content": message
    })

    # ✅ FAST MODEL
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=chat_sessions[session_id],
        temperature=0.4
    )

    reply = response.choices[0].message.content

    # Clean output
    reply = clean_text(reply)

    # Save response
    chat_sessions[session_id].append({
        "role": "assistant",
        "content": reply
    })

    return reply