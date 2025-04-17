import streamlit as st
import requests

st.title("🍳 Recipe Video Summarizer")
video_url = st.text_input("https://www.youtube.com/watch?v=XmQ8mZFqczw")

if st.button("Summarize Recipe"):
    with st.spinner("Processing video..."):
        response = requests.post("http://localhost:8000/process", json={"url": video_url})
        data = response.json()

        if data["success"]:
            st.success("Done!")
            st.markdown(data["recipe"])
        else:
            st.error(f"Error: {data['error']}")
