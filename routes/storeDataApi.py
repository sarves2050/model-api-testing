from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from db import storeData_collection
from fastapi.responses import JSONResponse
from typing import List

router = APIRouter()

class ChatEntry(BaseModel):
    prompt: str
    image_url: str

class SaveChatRequest(BaseModel):
    chat_data: List[ChatEntry]

@router.post("/chat")
async def save_chat(request_body: SaveChatRequest):
    """Create a new chat session and return chat_id"""
    
    chat_entries = [chat.dict() for chat in request_body.chat_data]
    
    chat_schema = {
        "chats": chat_entries
    }

    result = await storeData_collection.insert_one(chat_schema)
    new_chat_id = str(result.inserted_id)

    return JSONResponse(content={"message": "New chat saved successfully!", "chat_id": new_chat_id})

@router.get("/fetch-chat")
async def get_all_chats():
    """Fetch all chat data without chat_id (_id)"""
    chats_cursor = storeData_collection.find({}, {"_id": 0})  # Exclude _id field
    chats = await chats_cursor.to_list(length=None)  # Convert cursor to list

    return JSONResponse(content={"chats": chats})