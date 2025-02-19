import asyncio
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

app = FastAPI()

app.include_router(auth_router, prefix='/api/bitbee')
app.include_router(login_router, prefix='/api/bitbee')
app.include_router(chat_router, prefix='/api/bitbee')
app.include_router(store_router, prefix='/api/bitbee/random')
app.include_router(contact_router, prefix='/api/bitbee')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://bitbeeai.com", "https://www.bitbeeai.com" ,"https://testingbitbeeai.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await check_database_connection()

fine_model_path = 'bit0.1'

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_xl = StableDiffusionXLPipeline.from_pretrained(fine_model_path, torch_dtype=torch.float16).to(device)

class PromptRequest(BaseModel):
    user_id: str
    prompt: str = Field(..., max_length=50)  

generated_images: Dict[str, BytesIO] = {}

def calculate_sharpness(image: Image.Image) -> float:
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.var()

async def generate_image_async(pipe, prompt):
    loop = asyncio.get_event_loop()
    try:
        # Ensure the prompt is processed in the correct dtype
        return await loop.run_in_executor(None, lambda: pipe(prompt).images[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

semaphore = asyncio.Semaphore(2)  

@app.post("/api/images/generate")
async def generate_xl_image(request: PromptRequest):
    async with semaphore:
        try:
            # Convert prompt to float16 if necessary
            prompt = request.prompt
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(torch.float16)
            else:
                prompt = torch.tensor(prompt, dtype=torch.float16)

            image_xl = await generate_image_async(pipe_xl, prompt)
            sharpness_xl = calculate_sharpness(image_xl)

            img_byte_array = BytesIO()
            image_xl.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)

            generated_images[request.user_id] = img_byte_array

            headers = {"Sharpness": str(sharpness_xl), "Generated-By": "Main Model Bee"}
            return {"message": "Image generated successfully", "user_id": request.user_id}
        except HTTPException as e:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/api/images/retrieve/{user_id}")
async def retrieve_image(user_id: str):
    if user_id not in generated_images:
        raise HTTPException(status_code=404, detail="Image not found for the given user ID")

    img_byte_array = generated_images.pop(user_id)  
    return StreamingResponse(img_byte_array, media_type="image/png")

@app.get("/")
async def health_check():
    return {"status": "success", "message": "Jai Shree RAM BitbeeAI API is running!"}
