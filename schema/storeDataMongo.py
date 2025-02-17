from pydantic import BaseModel
from typing import List, Optional

# Chat Schema for saving chat history (without access_token in the individual chat entries)
class ChatEntry(BaseModel):
    prompt: str  # User's input prompt
    image_url: str   # URL to the generated image (optional)

class ChatSchema(BaseModel):
    
    
    chats: List[ChatEntry]  # List of chat entries
