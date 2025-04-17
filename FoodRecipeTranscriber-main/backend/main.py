from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

@app.post("/process")
async def process_video(request: VideoRequest):
    try:
        recipe = run_pipeline(request.url)
        return {"success": True, "recipe": recipe}
    except Exception as e:
        return {"success": False, "error": str(e)}
