import os
import sys
import subprocess
from typing import Optional

def setup_ffmpeg():
    """Ensure FFmpeg is available in the system path"""
    try:
        # Try running ffmpeg to check if it's available
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # If on Streamlit sharing, use the built-in binary path
        st.warning("FFmpeg not found in PATH - using fallback location")
        os.environ["PATH"] += os.pathsep + "/usr/local/bin"
        os.environ["FFMPEG_PATH"] = "/usr/local/bin/ffmpeg"
        
        # Also try installing ffmpeg-python as fallback
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"],
                          check=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            st.error("Could not install ffmpeg-python fallback")

