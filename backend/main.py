from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai_processor = GeminiProcessor(
    model_name="gemini-pro",
    project="gemini-dynamo-427100"
)

@app.post("/analyze_video")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        processor = YoutubeProcessor(genai_processor=genai_processor)
        result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose=True)

        key_concepts = processor.find_key_concepts(result, verbose=True)

        return key_concepts

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
