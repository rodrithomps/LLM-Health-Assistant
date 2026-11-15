from fastapi import APIRouter, UploadFile, File, HTTPException
import base64
from io import BytesIO
import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS

load_dotenv() 

router = APIRouter(prefix="/api/voice_chat", tags=["Voice Chat"])  # Using prefix and tags

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)

@router.post("/")
async def voice_chat(file: UploadFile = File(...)):
    """
    Analyze the user's audio with Gemini and return text with synthesized audio.
    """
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY is not configured")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    try:
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

        mime_type = file.content_type or "audio/wav"
        prompt = (
            "You are a health assistant. Understand the user's speech and answer "
            "concisely in the same language. Include a reminder that you are not "
            "a substitute for professional medical advice."
        )
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=audio_data, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=1024,
            ),
        )
        text_response = response.text
        if not text_response:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned no text for the uploaded audio.",
            )

        audio_buffer = BytesIO()
        gTTS(text=text_response, lang="en").write_to_fp(audio_buffer)
        audio_data_base64 = base64.b64encode(audio_buffer.getvalue()).decode("ascii")
        return {"response": text_response, "audio": audio_data_base64}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Voice chat processing failed")
        raise HTTPException(status_code=500, detail=f"Voice chat processing failed: {e}") from e
