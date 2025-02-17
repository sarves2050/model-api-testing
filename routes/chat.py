from fastapi import APIRouter, HTTPException, Request ,Depends
from pydantic import BaseModel
from db import chats_collection, users_collection
from fastapi.responses import JSONResponse
from typing import List, Optional
from bson import ObjectId



router = APIRouter()


class ChatEntry(BaseModel):
    prompt: str
    image_url: str

class SaveChatRequest(BaseModel):
    chat_data: List[ChatEntry]

async def get_access_token(request: Request) -> str:
    """Extract the access token from the Authorization header"""
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Access token required in Authorization header")

    return auth_header.split("Bearer ")[1]  # Extract token after "Bearer "

@router.post("/save-chat")
async def save_chat(
    request: Request, 
    request_body: SaveChatRequest, 
    access_token: str = Depends(get_access_token)
):
    """Create a new chat session and return chat_id"""

  

    user = await users_collection.find_one({"token": access_token})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    chat_entries = [chat.dict() for chat in request_body.chat_data]

    chat_schema = {
        "access_token": access_token,
        "chats": chat_entries
    }

    result = await chats_collection.insert_one(chat_schema)
    new_chat_id = str(result.inserted_id)

    return JSONResponse(content={"message": "New chat saved successfully!", "chat_id": new_chat_id})




@router.post("/save-chat/{chat_id}")
async def update_chat(
    chat_id: str, 
    request: Request, 
    request_body: SaveChatRequest, 
    access_token: str = Depends(get_access_token)
):
    """Append messages to an existing chat session using chat_id"""

    print(f"Received access_token: {access_token}")  # Debugging

    user = await users_collection.find_one({"token": access_token})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

    chat_obj_id = ObjectId(chat_id)

    existing_chat = await chats_collection.find_one({"_id": chat_obj_id, "access_token": access_token})
    if not existing_chat:
        raise HTTPException(status_code=404, detail="Chat ID not found")

    chat_entries_dict = [chat.dict() for chat in request_body.chat_data]

    await chats_collection.update_one(
        {"_id": chat_obj_id},
        {"$push": {"chats": {"$each": chat_entries_dict}}}
    )

    return JSONResponse(content={"message": "Chat updated successfully!", "chat_id": chat_id})




@router.get("/get-chat/{chat_id}")
async def get_chat(chat_id: str, request: Request, access_token: str = Depends(get_access_token)):
    """Retrieve chat history by chat_id"""

    if not ObjectId.is_valid(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

    chat_obj_id = ObjectId(chat_id)

    chat = await chats_collection.find_one({"_id": chat_obj_id, "access_token": access_token})

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat["_id"] = str(chat["_id"])  # Convert ObjectId to string

    return JSONResponse(content=chat)



@router.get("/get-first-prompts")
async def get_first_prompts(request: Request, access_token: str = Depends(get_access_token)):
    """Retrieve the first prompt along with the chat _id for each chat session of the authenticated user"""

    user = await users_collection.find_one({"token": access_token})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch all chat documents associated with the access token
    chats = await chats_collection.find({"access_token": access_token}).to_list(None)

    if not chats:
        raise HTTPException(status_code=404, detail="No chats found for this user")

    # Extract the _id and first prompt from each chat entry if available
    first_prompts = [
        {"_id": str(chat["_id"]), "prompt": chat["chats"][0]["prompt"]}
        for chat in chats if chat.get("chats") and len(chat["chats"]) > 0
    ]

    return JSONResponse(content={"first_prompts": first_prompts})
