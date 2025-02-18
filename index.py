from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from torch.cuda.amp import autocast  # Importing autocast for mixed precision
from io import BytesIO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from PIL import Image
import asyncio

from db import check_database_connection
from routes.apiSingup import router as auth_router
from routes.apiLogin import router as login_router  
from routes.chat import router as chat_router
from routes.storeDataApi import router as store_router
from routes.contactApi import router as contact_router

# Start backend cmd: python -m uvicorn index:app --host 0.0.0.0 --port 8000 --workers 1 --reload
app = FastAPI()

# Include routers
app.include_router(auth_router, prefix='/api/bitbee')
app.include_router(login_router, prefix='/api/bitbee')
app.include_router(chat_router, prefix='/api/bitbee')
app.include_router(store_router, prefix='/api/bitbee/random')
app.include_router(contact_router, prefix='/api/bitbee')

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173" , "https://bitbeeai.com" , "https://www.bitbeeai.com" ,"https://testingbitbeeai.netlify.app"],  
    allow_credentials=True,  
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await check_database_connection()

# Model loading path
fine_model_path = 'bit0.1'

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use the correct model (Stable Diffusion XL)
pipe_xl = StableDiffusionXLPipeline.from_pretrained(fine_model_path, torch_dtype=torch.float32).to(device)

class PromptRequest(BaseModel):
    prompt: str

def calculate_sharpness(image: Image.Image) -> float:
    """
    Calculate image sharpness using Laplacian Variance.
    """
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()

async def generate_image_async(pipe, prompt):
    """
    Run image generation in a separate thread to avoid blocking FastAPI’s event loop.
    """
    loop = asyncio.get_event_loop()

    # Use autocast to enable mixed precision (AMP)
    with autocast("cuda"):
        return await loop.run_in_executor(None, lambda: pipe(prompt).images[0])

@app.post("/api/images/generate")
async def generate_xl_image(request: PromptRequest):
    # Generate image asynchronously with mixed precision enabled
    image_xl = await generate_image_async(pipe_xl, request.prompt)
    
    # Calculate sharpness of the generated image
    sharpness_xl = calculate_sharpness(image_xl)

    # Convert image to byte array for response
    img_byte_array = BytesIO()
    image_xl.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)

    # Include sharpness info in response headers
    headers = {"Sharpness": str(sharpness_xl), "Generated-By": "Main Model Bee"}
    return StreamingResponse(img_byte_array, media_type="image/png", headers=headers)

@app.get("/")
async def health_check():
    """
    Health check endpoint to verify API status.
    """
    return {"status": "success", "message": " Jai Shree RAM BitbeeAI API is running!"}
