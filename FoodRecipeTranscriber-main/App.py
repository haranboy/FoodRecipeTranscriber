import streamlit as st
from pytube import YouTube
import subprocess
import whisper
import google.generativeai as genai
import os
from dotenv import load_dotenv
import google.api_core.exceptions as google_exceptions

# Load Gemini API key from Streamlit secrets or .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or .env file.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# Try reducing the maximum transcript length
MAX_TRANSCRIPT_CHARS = 10000  # Try a smaller limit first

def stream_audio_from_youtube(url, output_path="audio.mp3"):
    """Stream and download audio-only from YouTube instead of the full video"""
    try:
        yt = YouTube(url)
        # Get audio-only stream instead of video
        stream = yt.streams.filter(only_audio=True).first()
        
        # Download audio to a temporary file
        stream.download(filename=output_path)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to stream audio: {e}")

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def summarize_transcript(transcript):
    # Clean the transcript text to remove problematic characters
    transcript = transcript[:MAX_TRANSCRIPT_CHARS].strip()
    
    # Debug mode to inspect the transcript
    if st.session_state.get('debug_mode', False):
        st.text_area("Transcript sample (first 500 chars)", transcript[:500])
        st.write(f"Total transcript length: {len(transcript)} characters")
    
    prompt = f"""
    From the following cooking video transcript, extract:
    1. A list of ingredients.
    2. Step-by-step instructions.
    Transcript:
    {transcript}
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        # Add temperature and max output tokens parameters
        response = model.generate_content(
            prompt, 
            stream=False,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
        )
        return response.text.strip()
    except Exception as e:
        # Log more details about the error
        st.error(f"Gemini API error details: {type(e).__name__}: {str(e)}")
        # You could add debug logging here
        st.write(f"Prompt length was: {len(prompt)} characters")
        raise RuntimeError(f"Gemini API error: {str(e)}")

# Streamlit UI
st.title("🍳 Recipe Video Summarizer")
st.write("Enter a YouTube cooking video link to extract the ingredients and step-by-step instructions.")

# Add debug checkbox in the UI properly
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
st.session_state.debug_mode = st.checkbox("Debug mode", value=st.session_state.debug_mode)

video_url = st.text_input("Your Video URL Link")
if video_url and st.button("Summarize Recipe"):
    try:
        with st.spinner("Streaming audio from YouTube..."):
            audio_path = stream_audio_from_youtube(video_url)
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_path)
        with st.spinner("Summarizing recipe with Gemini..."):
            summary = summarize_transcript(transcript)
        st.success("Recipe Summary Ready!")
        st.markdown(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
