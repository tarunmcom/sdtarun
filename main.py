from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import torch
from diffusers import DiffusionPipeline
import base64
from io import BytesIO
import random
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
try:
    logger.info(f"Loading model: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Enable CPU offloading
try:
    logger.info("Enabling CPU offloading")
    pipe.enable_model_cpu_offload()
    logger.info("CPU offloading enabled")
except Exception as e:
    logger.error(f"Error enabling CPU offloading: {str(e)}")
    raise

# Load LoRA weights
lora_weights_path = "/app/models/pytorch_lora_weights.safetensors"
if os.path.exists(lora_weights_path):
    try:
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        pipe.load_lora_weights(lora_weights_path)
        pipe.fuse_lora()
        logger.info("LoRA weights loaded and fused successfully")
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {str(e)}")
        raise
else:
    logger.warning(f"LoRA weights file not found at {lora_weights_path}")

class ImageRequest(BaseModel):
    positive_prompt: str
    negative_prompt: str
    num_steps: int = 55
    num_images: int = 1
    height: int = 1024
    width: int = 1024

class ImageResponse(BaseModel):
    images: List[str]
    seeds: List[int]

@app.post("/generate", response_model=ImageResponse)
async def generate_images(request: ImageRequest):
    try:
        logger.info(f"Received image generation request: {request}")
        images = []
        seeds = []

        for i in range(request.num_images):
            logger.info(f"Generating image {i+1}/{request.num_images}")
            seed = random.randint(0, 4294967295)  # Full range of 32-bit integer
            logger.info(f"Using seed: {seed}")
            
            try:
                generator = torch.Generator("cuda").manual_seed(seed)
                logger.info("Created CUDA generator")
            except Exception as e:
                logger.error(f"Error creating CUDA generator: {str(e)}")
                raise

            try:
                logger.info("Calling pipe for image generation")
                image = pipe(
                    prompt=request.positive_prompt,
                    negative_prompt=request.negative_prompt,
                    generator=generator,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_steps
                ).images[0]
                logger.info("Image generated successfully")
            except Exception as e:
                logger.error(f"Error during image generation: {str(e)}")
                raise

            try:
                logger.info("Encoding image to base64")
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                images.append(base64.b64encode(buffered.getvalue()).decode())
                seeds.append(seed)
                logger.info("Image encoded and added to response")
            except Exception as e:
                logger.error(f"Error encoding image: {str(e)}")
                raise

        logger.info("All images generated successfully")
        return ImageResponse(images=images, seeds=seeds)
    except Exception as e:
        logger.error(f"Unhandled exception in generate_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Welcome to Tarun_qwer</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 p-8">
            <div class="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl p-6">
                <h1 class="text-2xl font-bold mb-4">Welcome to Tarun_qwer</h1>
                <nav>
                    <ul class="space-y-2">
                        <li>
                            <a href="/ui" class="text-blue-600 hover:text-blue-800">Image Generation UI</a>
                        </li>
                        <li>
                            <a href="/docs" class="text-blue-600 hover:text-blue-800">API Documentation</a>
                        </li>
                    </ul>
                </nav>
            </div>
        </body>
    </html>
    """

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """
    <html>
        <head>
            <title>SDXL LoRA Image Generator</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 p-8">
            <div class="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl p-6">
                <h1 class="text-2xl font-bold mb-4">SDXL LoRA Image Generator</h1>
                <form id="generate-form" class="space-y-4">
                    <div>
                        <label for="positive_prompt" class="block text-sm font-medium text-gray-700">Positive Prompt</label>
                        <input type="text" id="positive_prompt" name="positive_prompt" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="negative_prompt" class="block text-sm font-medium text-gray-700">Negative Prompt</label>
                        <input type="text" id="negative_prompt" name="negative_prompt" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="num_steps" class="block text-sm font-medium text-gray-700">Number of Steps</label>
                        <input type="number" id="num_steps" name="num_steps" value="55" min="1" max="100" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="num_images" class="block text-sm font-medium text-gray-700">Number of Images</label>
                        <input type="number" id="num_images" name="num_images" value="1" min="1" max="4" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="height" class="block text-sm font-medium text-gray-700">Height</label>
                        <input type="number" id="height" name="height" value="1024" min="64" max="2048" step="64" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="width" class="block text-sm font-medium text-gray-700">Width</label>
                        <input type="number" id="width" name="width" value="1024" min="64" max="2048" step="64" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <button type="submit" class="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">Generate Images</button>
                </form>
                <div id="results" class="mt-6 grid grid-cols-2 gap-4"></div>
            </div>
            <script>
                document.getElementById('generate-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    const data = Object.fromEntries(formData.entries());
                    data.num_steps = parseInt(data.num_steps);
                    data.num_images = parseInt(data.num_images);
                    data.height = parseInt(data.height);
                    data.width = parseInt(data.width);

                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data),
                    });

                    const result = await response.json();
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = '';
                    result.images.forEach((image, index) => {
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${image}`;
                        img.alt = `Generated Image ${index + 1}`;
                        img.className = 'w-full h-auto rounded-lg shadow-md';
                        resultsDiv.appendChild(img);
                    });
                });
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
