import asyncio
import torch
import cv2
import numpy as np
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from typing import Dict


from db import check_database_connection
from routes.apiSingup import router as auth_router
from routes.apiLogin import router as login_router  
from routes.chat import router as chat_router
from  routes.storeDataApi import router as store_router
from routes.contactApi import router as contact_router


# Initialize FastAPI App
app = FastAPI()

app.include_router(auth_router, prefix='/api/bitbee')
app.include_router(login_router, prefix='/api/bitbee')
app.include_router(chat_router, prefix='/api/bitbee')
app.include_router(store_router, prefix='/api/bitbee/random')
app.include_router(contact_router, prefix='/api/bitbee')

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://bitbeeai.com", "https://testingbitbeeai.netlify.app" ,  "https://api.bitbeeai.com" ,"https://www.bitbeeai.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await check_database_connection()

# Model Path
fine_model_path = 'bit0.1'

# Check Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use bfloat16 for NVIDIA L4 GPU
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Load Model with Explicit bfloat16
pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    fine_model_path, torch_dtype=torch_dtype
).to(device)
pipe_xl.to(device, dtype=torch_dtype)  # Ensure full model uses bfloat16

# Request Model
class PromptRequest(BaseModel):
    user_id: str
    prompt: str = Field(..., max_length=50)    # Limit prompt length to 50 characters

# In-memory store for generated images
generated_images: Dict[str, BytesIO] = {}

# Image Sharpness Calculation
def calculate_sharpness(image: Image.Image) -> float:
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()

# Image Generation Function
async def generate_image(pipe, prompt):
    """Run the Stable Diffusion pipeline asynchronously with autocast."""
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # Enable mixed precision
            result = await asyncio.to_thread(lambda: pipe(prompt).images[0])  # Fix dtype issue
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# Semaphore to Limit Concurrent Requests
semaphore = asyncio.Semaphore(1)  # Allows 3 parallel generations

# Image Generation API
@app.post("/api/images/generate")
async def generate_xl_image(request: PromptRequest):
    async with semaphore:
        try:
            image_xl = await generate_image(pipe_xl, request.prompt)
            sharpness_xl = calculate_sharpness(image_xl)

            img_byte_array = BytesIO()
            image_xl.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)

            # Store the image in memory with user_id as the key
            generated_images[request.user_id] = img_byte_array

            # Free GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return {"message": "Image generated successfully", "user_id": request.user_id, "sharpness": sharpness_xl}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Retrieve Image API
@app.get("/api/images/retrieve/{user_id}")
async def retrieve_image(user_id: str):
    if user_id not in generated_images:
        raise HTTPException(status_code=404, detail="Image not found for the given user ID")

    img_byte_array = generated_images.pop(user_id)  # Remove after retrieval
    return StreamingResponse(img_byte_array, media_type="image/png")

# Health Check API
@app.get("/")
async def health_check():
    return {"status": "success", "message": "Jai Shree RAM 🚀 BitbeeAI API is running!"}
