# === app.py (Streamlit Cloud-ready) ===
import streamlit as st
from pytube import YouTube
import subprocess
import whisper
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Gemini API key from Streamlit secrets or .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

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

def summarize_transcript(transcript):
    prompt = f"""
    From the following cooking video transcript, extract:
    1. A list of ingredients.
    2. Step-by-step instructions.

    Transcript:
    {transcript}
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.title("🍳 Recipe Video Summarizer")
video_url = st.text_input("Paste a YouTube cooking video link:")

if st.button("Summarize Recipe"):
    try:
        with st.spinner("Downloading video..."):
            video_path = download_video(video_url)
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path)
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(audio_path)
        with st.spinner("Summarizing recipe with Gemini..."):
            summary = summarize_transcript(transcript)

        st.success("Done!")
        st.markdown(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
