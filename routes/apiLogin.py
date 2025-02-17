from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from passlib.context import CryptContext
from db import users_collection
import os
from fastapi.responses import JSONResponse

# Pydantic Model for Login Request
class LoginRequest(BaseModel):
    email: str
    password: str

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

router = APIRouter()

@router.post("/login")
async def login(request: LoginRequest, response: Response):  # Accepting data as request body
    """Login API: Check if email exists, verify password, and store JWT token in cookies."""
    user = await users_collection.find_one({"email": request.email})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not user["verified"]:
        raise HTTPException(status_code=400, detail="User not verified. Please verify your email first.")

    if not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password.")
    
    # Retrieve the existing JWT token from the database
    token = user.get("token")

    if not token:
        raise HTTPException(status_code=400, detail="No token found. Please verify your email first.")

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=False,  # Prevents JS from accessing it
        samesite="Lax",  # Change to "None" if frontend and backend are on different domains
        secure=False  # Change to True in production with HTTPS
    )


    return JSONResponse(content={"message": "Login successful!", "token": token})
