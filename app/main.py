from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
from datetime import datetime
import sys
import uuid

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_sentiment
from utils.ai_response import get_ai_response

app = FastAPI(title="Citizen AI API", description="Backend API for Citizen AI platform")

# Data Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None

class FeedbackItem(BaseModel):
    text: str
    category: Optional[str] = None
    user_id: Optional[str] = None

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float
    timestamp: str
    id: str

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)
feedback_file = os.path.join(data_dir, "feedback.json")

# Initialize feedback file if it doesn't exist
if not os.path.exists(feedback_file):
    with open(feedback_file, "w") as f:
        json.dump([], f)

# Helper function to load feedback data
def load_feedback_data():
    try:
        with open(feedback_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Helper function to save feedback data
def save_feedback_data(data):
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)

@app.get("/")
async def root():
    return {"message": "Welcome to Citizen AI API"}

@app.post("/chat", response_model=Dict[str, Any])
async def chat(chat_message: ChatMessage):
    try:
        response = await get_ai_response(chat_message.message, chat_message.session_id, chat_message.context)
        return {"response": response, "session_id": chat_message.session_id or str(uuid.uuid4())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=SentimentResponse)
async def submit_feedback(feedback: FeedbackItem):
    # Analyze sentiment
    sentiment_result = analyze_sentiment(feedback.text)
    
    # Create response with timestamp and ID
    timestamp = datetime.now().isoformat()
    feedback_id = str(uuid.uuid4())
    
    # Create response object
    response = {
        "id": feedback_id,
        "text": feedback.text,
        "category": feedback.category,
        "user_id": feedback.user_id,
        "sentiment": sentiment_result["sentiment"],
        "score": sentiment_result["score"],
        "timestamp": timestamp
    }
    
    # Save to file
    data = load_feedback_data()
    data.append(response)
    save_feedback_data(data)
    
    return response

@app.get("/feedback", response_model=List[Dict[str, Any]])
async def get_feedback():
    return load_feedback_data()

@app.get("/sentiment/summary")
async def get_sentiment_summary():
    data = load_feedback_data()
    
    if not data:
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    
    # Count sentiments
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for item in data:
        sentiment = item.get("sentiment", "neutral").lower()
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    
    # Add total
    sentiment_counts["total"] = len(data)
    
    return sentiment_counts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)