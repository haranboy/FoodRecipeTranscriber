import streamlit as st
from pytube import YouTube
import subprocess
import whisper
import google.generativeai as genai
import os
from dotenv import load_dotenv
import google.api_core.exceptions as google_exceptions

# Add this before the API call to debug
if st.checkbox("Debug mode"):
    st.text_area("Transcript sample (first 500 chars)", transcript[:500])
    st.write(f"Total transcript length: {len(transcript)} characters")
    
# Load Gemini API key from Streamlit secrets or .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Try reducing the maximum transcript length
MAX_TRANSCRIPT_CHARS = 10000  # Try a smaller limit first

def download_video(url, output_path="video.mp4"):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
    stream.download(filename=output_path)
    return output_path

def extract_audio(video_path, audio_path="audio.mp3"):
    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Modify your summarize_transcript function to better handle the API request
def summarize_transcript(transcript):
    # Clean the transcript text to remove problematic characters
    transcript = transcript[:MAX_TRANSCRIPT_CHARS].strip()
    
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

video_url = st.text_input("Your Video URL Link")

if video_url and st.button("Summarize Recipe"):
    try:
        with st.spinner("Downloading video..."):
            video_path = download_video(video_url)
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path)
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_path)
        with st.spinner("Summarizing recipe with Gemini..."):
            summary = summarize_transcript(transcript)

        st.success("Recipe Summary Ready!")
        st.markdown(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
