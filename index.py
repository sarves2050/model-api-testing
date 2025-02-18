import torch
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
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

app = FastAPI()

# ✅ Suppress PyTorch Dynamo Graph Break Warnings
torch._dynamo.config.suppress_errors = True

# ✅ Enable TensorFloat32 Precision for Better GPU Performance
torch.set_float32_matmul_precision('high')

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://bitbeeai.com", "https://www.bitbeeai.com", "https://testingbitbeeai.netlify.app"],  
    allow_credentials=True,  
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
    allow_headers=["*"],
)

# ✅ Check CUDA Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 CUDA Available! Using GPU: {gpu_name}")
else:
    print("⚠️ CUDA not available. Using CPU!")

# ✅ Load Fine-tuned Model
fine_model_path = 'bit0.1'
pipe_xl = StableDiffusionXLPipeline.from_pretrained(fine_model_path, torch_dtype=dtype).to(device)

# ✅ Compile Model for Faster Inference (Only for PyTorch 2.0+)
if torch.__version__ >= "2.0" and device.type == "cuda":
    pipe_xl = torch.compile(pipe_xl)

# ✅ Register API Routes
app.include_router(auth_router, prefix="/api")
app.include_router(login_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(store_router, prefix="/api")
app.include_router(contact_router, prefix="/api")

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
    Generate an image asynchronously.
    """
    loop = asyncio.get_event_loop()
    
    # ✅ Correct autocast usage for PyTorch
    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        return await loop.run_in_executor(None, lambda: pipe(prompt).images[0])

@app.post("/api/images/generate")
async def generate_xl_image(request: PromptRequest):
    """
    Endpoint to generate an image from a text prompt.
    """
    try:
        image_xl = await generate_image_async(pipe_xl, request.prompt)
        sharpness_xl = calculate_sharpness(image_xl)

        img_byte_array = BytesIO()
        image_xl.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)

        headers = {"Sharpness": str(sharpness_xl), "Generated-By": "Main Model Bee"}
        return StreamingResponse(img_byte_array, media_type="image/png", headers=headers)
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def health_check():
    """
    Health check endpoint to verify API status.
    """
    return {"status": "success", "message": "🚀 Jai Shree RAM! BitBeeAI API is running!"}
