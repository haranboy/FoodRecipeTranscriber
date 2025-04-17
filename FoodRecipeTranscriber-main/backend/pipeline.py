import subprocess
import whisper
from pytube import YouTube
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("YOUR_GEMINI_KEY"))

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

def run_pipeline(video_url):
    video_path = download_video(video_url)
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    summary = summarize_transcript(transcript)
    return summary
