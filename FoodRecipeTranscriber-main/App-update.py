import streamlit as st
import subprocess
import whisper
import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv
import re
from typing import Optional

def setup_ffmpeg():
    ffmpeg_path = os.getenv("FFMPEG_PATH", "/usr/local/bin/ffmpeg")  # Define ffmpeg_path
    output_path = "audio.webm"  # Define output_path as a placeholder
    url = "https://www.youtube.com/watch?v=5IwmuaKE7tA"  # Placeholder URL for demonstration

    os.environ["FFMPEG_PATH"] = ffmpeg_path
    os.environ["FFPROBE_PATH"] = "/usr/local/bin/ffprobe"

    command = [
    "yt-dlp",
    "--ffmpeg-location", ffmpeg_path,
    "-x",  # Extract audio
    "--audio-format", "best",  # Let yt-dlp choose best format
    "--output", output_path,
    url,
    ]
    
    """Ensure FFmpeg is available in the system path"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        st.warning("FFmpeg not found in PATH - using fallback location")
        os.environ["PATH"] += os.pathsep + "/usr/local/bin"
        os.environ["FFMPEG_PATH"] = "/usr/local/bin/ffmpeg"
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"],
                          check=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            st.error("Could not install ffmpeg-python fallback")

# Call this right after your Streamlit imports
setup_ffmpeg()

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

# Add this near the top of your imports
import os
import sys

def ensure_ffmpeg():
    """Ensure FFmpeg is available in the system path"""
    try:
        # Try importing ffmpeg to check if it's available
        import ffmpeg
    except ImportError:
        # Install ffmpeg-python if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
    
    # Check if ffmpeg binary is available
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
    except FileNotFoundError:
        # If on Streamlit sharing, use the built-in binary
        st.warning("FFmpeg not found - using fallback method")
        os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# Call this function before any Whisper operations
ensure_ffmpeg()

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
    """Download audio using yt-dlp with proper format handling"""
    try:
        # Install yt-dlp if not available
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Use yt-dlp to download best audio and convert to webm
        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "best",  # Let yt-dlp choose best format
            "--output", output_path,
            url,
        ]

        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not os.path.exists(output_path):
            # Try alternative approach if direct download fails
            temp_path = "temp_audio"
            command = [
                "yt-dlp",
                "-x",
                "--audio-format", "best",
                "--output", temp_path,
                url,
            ]
            subprocess.run(command, check=True)
            
            # Find the downloaded file and rename it
            for f in os.listdir():
                if f.startswith(temp_path):
                    os.rename(f, output_path)
                    break

        if not os.path.exists(output_path):
            raise FileNotFoundError("Download completed but file not found")

        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = f"Download failed: {e.stderr}" if e.stderr else "Unknown download error"
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(f"Download error: {str(e)}") from e
        
def transcribe_audio(audio_path):
    # Ensure FFmpeg is available
    ensure_ffmpeg()
    
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        
        # Add explicit FFmpeg path for Whisper
        import whisper.audio
        whisper.audio.ffmpeg_path = "ffmpeg"  # or "/usr/local/bin/ffmpeg" if needed
        
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
