import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from diffusers import StableDiffusionXLPipeline
import torch
from io import BytesIO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from PIL import Image
from typing import Dict
from db import check_database_connection
from routes.apiSingup import router as auth_router
from routes.apiLogin import router as login_router
from routes.chat import router as chat_router
from routes.storeDataApi import router as store_router
from routes.contactApi import router as contact_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Include Routers
app.include_router(auth_router, prefix='/api/bitbee')
app.include_router(login_router, prefix='/api/bitbee')
app.include_router(chat_router, prefix='/api/bitbee')
app.include_router(store_router, prefix='/api/bitbee/random')
app.include_router(contact_router, prefix='/api/bitbee')

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://bitbeeai.com", "https://www.bitbeeai.com", "https://testingbitbeeai.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_db():
    await check_database_connection()

# Model path
fine_model_path = 'bit0.1'

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if torch.cuda.is_available() else torch.float32

# Load the model with optimizations
try:
    logger.info("Loading Stable Diffusion XL Model...")
    pipe_xl = StableDiffusionXLPipeline.from_pretrained(
        fine_model_path, 
        torch_dtype=dtype,
    ).to(device)

    # Enable performance optimizations
    pipe_xl.enable_xformers_memory_efficient_attention()  # Speeds up inference
    pipe_xl.enable_model_cpu_offload()  # Moves unused parts to CPU to reduce GPU usage
    pipe_xl.enable_attention_slicing()  # Lowers memory consumption

    logger.info("Stable Diffusion XL Model Loaded Successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load Stable Diffusion model")

# Request Model
class PromptRequest(BaseModel):
    user_id: str
    prompt: str = Field(..., max_length=100)

# Store generated images in memory
generated_images: Dict[str, BytesIO] = {}

# Calculate Image Sharpness
def calculate_sharpness(image: Image.Image) -> float:
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()

# Image generation function
async def generate_image_async(pipe, prompt):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: pipe(prompt).images[0])
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(1)

@app.post("/api/images/generate")
async def generate_xl_image(request: PromptRequest):
    async with semaphore:
        try:
            image_xl = await generate_image_async(pipe_xl, request.prompt)
            sharpness_xl = calculate_sharpness(image_xl)

            img_byte_array = BytesIO()
            image_xl.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)

            generated_images[request.user_id] = img_byte_array

            logger.info(f"Generated image for {request.user_id} with sharpness: {sharpness_xl}")
            return {"message": "Image generated successfully", "user_id": request.user_id}
        except HTTPException as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/api/images/retrieve/{user_id}")
async def retrieve_image(user_id: str):
    if user_id not in generated_images:
        raise HTTPException(status_code=404, detail="Image not found for the given user ID")

    img_byte_array = generated_images.pop(user_id)
    return StreamingResponse(img_byte_array, media_type="image/png")

@app.get("/")
async def health_check():
    return {"status": "success", "message": "BitbeeAI API is running!"}
