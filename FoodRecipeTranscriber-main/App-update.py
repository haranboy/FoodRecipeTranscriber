import streamlit as st
import subprocess
import whisper
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from typing import Optional

# Configuration
load_dotenv()
MAX_TRANSCRIPT_CHARS = 10000
SUPPORTED_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MODEL = "base"

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or .env file.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_whisper_model(model_size: str = DEFAULT_MODEL):
    """Load and cache the Whisper model"""
    if model_size not in SUPPORTED_WHISPER_MODELS:
        raise ValueError(f"Unsupported model size. Choose from: {SUPPORTED_WHISPER_MODELS}")
    return whisper.load_model(model_size)

def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return re.match(youtube_regex, url) is not None

def download_audio_with_ytdlp(url: str, output_path: str = "audio.webm") -> str:
    """Download audio using yt-dlp"""
    try:
        subprocess.run(["pip", "install", "yt-dlp"], check=True, capture_output=True)
        
        command = [
            "yt-dlp", 
            "-f", "bestaudio",
            "--no-playlist",
            "-o", output_path,
            "--extract-audio",
            "--audio-format", "webm",
            url
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        if not os.path.exists(output_path):
            raise FileNotFoundError("Download completed but file not found")
            
        return output_path
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e.stderr}") from e
    except Exception as e:
        raise RuntimeError(f"Download error: {str(e)}") from e

def transcribe_audio(audio_path: str, model_size: str = DEFAULT_MODEL) -> str:
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper_model(model_size)
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}") from e

def summarize_transcript(transcript: str) -> str:
    """Summarize transcript using Gemini"""
    transcript = transcript[:MAX_TRANSCRIPT_CHARS].strip()
    
    if st.session_state.get('debug_mode', False):
        with st.expander("Debug: Transcript Sample"):
            st.text_area("First 500 characters", transcript[:500], height=200)
            st.write(f"Total length: {len(transcript)} characters")
    
    prompt = f"""
    Extract the following from this cooking video transcript:
    1. A clear list of ingredients with measurements
    2. Detailed step-by-step instructions
    3. Cooking time and difficulty level if mentioned
    
    Format the response using markdown with clear headings.
    
    Transcript:
    {transcript}
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # Lower for more factual responses
                "max_output_tokens": 2048,
            }
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        raise

def main():
    st.title("🍳 Recipe Video Summarizer")
    st.write("Extract ingredients and instructions from cooking videos")
    
    # Settings sidebar
    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox(
            "Whisper Model Size",
            SUPPORTED_WHISPER_MODELS,
            index=SUPPORTED_WHISPER_MODELS.index(DEFAULT_MODEL),
            help="Larger models are more accurate but slower"
        )
        st.session_state.debug_mode = st.checkbox(
            "Debug mode",
            value=st.session_state.get('debug_mode', False)
        )
    
    # Main interface
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a link to a cooking video"
    )
    
    if st.button("Summarize Recipe"):
        if not video_url:
            st.warning("Please enter a YouTube URL")
            return
            
        if not is_valid_youtube_url(video_url):
            st.error("Please enter a valid YouTube URL")
            return
            
        try:
            with st.spinner("Downloading audio..."):
                audio_path = download_audio_with_ytdlp(video_url)
                
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path, model_size)
                
            with st.spinner("Summarizing recipe..."):
                summary = summarize_transcript(transcript)
                
            st.success("Done!")
            st.markdown(summary)
            
            # Cleanup
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
