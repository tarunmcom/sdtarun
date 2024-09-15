from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import is_xformers_available
import base64
from io import BytesIO
import random
import os
import logging
from diffusers import DPMSolverMultistepScheduler
from safetensors.torch import load_file
from openai import OpenAI
import re
import json
######################################################################################################
#LLM Call
def make_prompt_list(n_images, prompts):
  prompts = prompts["prompts"]
  refined_prompts = []
  for i in range(n_images):
    refined_prompts.append(prompts[random.randint(0,len(prompts)-1)]["prompt"])
  return refined_prompts
    
def convert_to_json(input_text):
  # Check if the input matches the expected format
  pattern = r'^```\n\{\s*"prompts"\s*:\s*\[.*?\]\s*\}\s*```$'
  if not re.match(pattern, input_text, re.DOTALL):
    raise ValueError(
      "Invalid input format. Expected format:\n"
      '```\n{\n  "prompts": ["prompt1", "prompt2", ...]\n}\n```'
    )

  # Extract the JSON content from between the backticks
  json_content = re.search(r'```\n(.*)\n```', input_text, re.DOTALL).group(1)

  try:
    # Parse the JSON content
    data = json.loads(json_content)
  except json.JSONDecodeError:
    raise ValueError("Invalid JSON format within the backticks.")

  # Check if the parsed data has the expected structure
  if not isinstance(data, dict) or "prompts" not in data or not isinstance(data["prompts"], list):
    raise ValueError(
      "Invalid JSON structure. Expected a dictionary with a 'prompts' key containing a list."
    )

  # Create the new JSON structure
  result = {"prompts": [{"prompt": prompt} for prompt in data["prompts"]]}

  # Convert the result to a JSON string with indentation
  return json.dumps(result, indent=2)

api_key = os.getenv('API_KEY')
client = OpenAI(api_key=api_key)
def generate_llm_response(crude_prompt):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": "system_instruction=\"You are a helpful Prompt Engineer. You are tasked to refine user's input prompt for image generation using stable diffusion model version SDXL.The SDXL model generates good images if only the human subject is mostly visible and there is no or minimal background. If you try to generate wide angle shot or too many subjects in the image, then SDXL gives poor response, so you should avoid this by refining the prompt. You should refine the prompts in such a way that it follows users input prompt but the main human surject must occupy the maximum space in the image and background is minimal or blurred. You should also utilize the sdxl prompt generation skills by providing shot description, camera and lenses parameters so that you generate appealing images. You will receive the name of a person in encoded form and you should use that in your prompt. Example:\\n'''\\nUser: {\\\"prpmot\\\":\\\"I am riding a horse\\\", \\\"name\\\":\\\"Tarun_qwer\\\"} \\nrefined prompt:- \\\"A close-up of Tarun_qwer on a horse, with focus on the detailed subject while the background is blurred.Tarun_qwer is in riding attire, and the horse is in motion, emphasizing their expressions and textures, Tarun_qwer has proper eyes\\\"\\n'''\\nNever use wide angle or long distance shots. You may use the following shots 1. close up shot 2. medium shot 3. portrait. You may also use lenses focus and parameters that generate these type of shots.\\nAlso add relevant info so that face is generated well. And generate 4-5 different prompts from one prompt.\\nAlways generate prompt in 50-60 words because above that response is not good.\\nMake sure you give output in the following format:-\\n'''\\n {\\\"prompts\\\":[\\\"<PROMPT>\\\", \\\"<PROMPT>\\\",....]}\\n'''\\nYou must always generate prompts so that human subject occupies the maximum area in the image and more detailing on human face. You must add relevant face expression of the human subject more, so that face detailing is good.\","
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": crude_prompt
          }
        ]
      },
    ],
    temperature=0.9,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
      "type": "text"
    }
  )
  res = response.to_dict()
  textdata = (res["choices"][0]['message']['content'])
  return textdata
#LLM END
#######################################################################################

print("cuda v:", torch.version.cuda)
current_lora_weights = None
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

if is_xformers_available():
    pipe.enable_xformers_memory_efficient_attention()
    logger.info("enable_xformers_memory_efficient_attention successfully")
else:
    logger.warning("xformers is not available. Consider installing it for faster inference.")

# Global variable to store the current LoRA weights
current_lora_weights = None


class ImageRequest(BaseModel):
    positive_prompt: str
    negative_prompt: str
    num_steps: int = 40
    num_images: int = 1
    height: int = 1024
    width: int = 1024


class ImageResponse(BaseModel):
    images: List[str]
    seeds: List[int]


@app.post("/generate", response_model=ImageResponse)
async def generate_images(request: ImageRequest):
    global lora_user_name
    try:
        logger.info(f"Received image generation request: {request}")

        # Generate seeds for all images at once
        seeds = [random.randint(0, 4294967295) for _ in range(request.num_images)]
        logger.info(f"Using seeds: {seeds}")

        try:
            generators = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]
            logger.info("Created CPU generators")
        except Exception as e:
            logger.error(f"Error creating CPU generators: {str(e)}")
            raise

        try:
            logger.info("Making refined prompt start")
            crude_prompt = f"'prompt':{request.positive_prompt}, 'name':{lora_user_name}"
            logger.info(f"crude prompt is:- {crude_prompt}")
            response = generate_llm_response(crude_prompt)
            response_json = json.loads(convert_to_json(response))
            refined = make_prompt_list(request.num_images, response_json)
            logger.info(f"{refined}")
        except Exception as e:
            logger.error(f"Error while refining prompt : {e}")
            raise


        try:
            logger.info("Calling pipe for batch image generation")
            batch_images = pipe(
                prompt=refined,
                negative_prompt=[request.negative_prompt] * request.num_images,
                generator=generators,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_steps,
                num_images_per_prompt=1
            ).images
            logger.info("Images generated successfully")
        except Exception as e:
            logger.error(f"Error during batch image generation: {str(e)}")
            raise

        images = []
        try:
            logger.info("Encoding images to base64")
            for image in batch_images:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                images.append(base64.b64encode(buffered.getvalue()).decode())
            logger.info("Images encoded and added to response")
        except Exception as e:
            logger.error(f"Error encoding images: {str(e)}")
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
            <title>Welcome to SDXL LoRA Image Generator</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 p-8">
            <div class="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl p-6">
                <h1 class="text-2xl font-bold mb-4">Welcome to SDXL LoRA Image Generator</h1>
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

lora_user_name = "anonymous"

@app.post("/upload_lora")
async def upload_lora(file: UploadFile = File(...)):
    global current_lora_weights, pipe , lora_user_name
    try:
        logger.info(f"Received LoRA upload: {file.filename}")
        parts = file.filename.split('_')
        lora_user_name = '_'.join(parts[:2]).capitalize()
        logger.info(f"lora subject name is :{lora_user_name}")

        # Save the uploaded file
        lora_path = f"uploaded_lora_{file.filename}"
        contents = await file.read()
        with open(lora_path, "wb") as buffer:
            buffer.write(contents)

        # Unload previous LoRA if exists
        if current_lora_weights is not None:
            logger.info("Unloading previous LoRA weights")
            pipe.unload_lora_weights()

        # Load and fuse new LoRA weights
        logger.info(f"Loading new LoRA weights from: {lora_path}")
        pipe.load_lora_weights(lora_path)

        # Store the new LoRA weights
        current_lora_weights = lora_path

        logger.info("New LoRA weights loaded successfully")
        return {"message": "LoRA uploaded and applied successfully"}
    except Exception as e:
        logger.error(f"Error uploading LoRA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """
    <html>
        <head>
            <title>Text-to-Portrait AI</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .loader {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #3498db;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .fullscreen {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background-color: rgba(0, 0, 0, 0.9);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                }
                .fullscreen img {
                    max-width: 90%;
                    max-height: 90%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body class="bg-gray-100 p-8">
            <div class="max-w-3xl mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl p-6">
                <h1 class="text-2xl font-bold mb-4">Text-to-Portrait AI</h1>
                <div id="active-lora" class="mb-4 text-sm font-medium text-green-600"></div>
                <form id="lora-upload-form" enctype="multipart/form-data" class="mb-6">
                    <div class="mb-4">
                        <label for="lora-file" class="block text-sm font-medium text-gray-700">Upload LoRA File</label>
                        <input type="file" id="lora-file" name="file" accept=".safetensors" class="mt-1 block w-full">
                    </div>
                    <button type="submit" id="upload-lora-button" class="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500" disabled>Upload LoRA</button>
                </form>
                <div id="lora-status" class="mb-4 text-sm font-medium"></div>
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
                        <input type="number" id="num_images" name="num_images" value="1" min="1" max="32" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="height" class="block text-sm font-medium text-gray-700">Height</label>
                        <input type="number" id="height" name="height" value="1024" min="64" max="2048" step="64" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <div>
                        <label for="width" class="block text-sm font-medium text-gray-700">Width</label>
                        <input type="number" id="width" name="width" value="1024" min="64" max="2048" step="64" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                    </div>
                    <button type="submit" id="generate-button" class="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">Generate Images</button>
                </form>
                <div id="loading" class="hidden mt-4 flex justify-center items-center">
                    <div class="loader mr-3"></div>
                    <p>Processing... This may take a while.</p>
                </div>
                <div id="results" class="mt-6 grid grid-cols-2 gap-4"></div>
            </div>
            <div id="fullscreen-view" class="fullscreen hidden">
                <img id="fullscreen-image" src="" alt="Full screen image">
            </div>
            <script>
                const loraFileInput = document.getElementById('lora-file');
                const uploadLoraButton = document.getElementById('upload-lora-button');

                loraFileInput.addEventListener('change', () => {
                    uploadLoraButton.disabled = !loraFileInput.files.length;
                });

                document.getElementById('lora-upload-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    const loraFile = formData.get('file');
                    if (!loraFile) {
                        alert('Please select a LoRA file to upload.');
                        return;
                    }

                    document.getElementById('loading').classList.remove('hidden');
                    document.getElementById('generate-button').disabled = true;
                    uploadLoraButton.disabled = true;

                    try {
                        const response = await fetch('/upload_lora', {
                            method: 'POST',
                            body: formData
                        });

                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail || 'Unknown error'}`);
                        }

                        const result = await response.json();
                        document.getElementById('lora-status').textContent = result.message;

                        // Update the active LoRA display
                        document.getElementById('active-lora').textContent = `Active LoRA: ${loraFile.name}`;

                        // Clear the file input
                        loraFileInput.value = '';
                        uploadLoraButton.disabled = true;
                    } catch (error) {
                        console.error('Error:', error);
                        alert(`An error occurred while uploading the LoRA file: ${error.message}`);
                    } finally {
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('generate-button').disabled = false;
                    }
                });

                document.getElementById('generate-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    const data = Object.fromEntries(formData.entries());
                    data.num_steps = parseInt(data.num_steps);
                    data.num_images = parseInt(data.num_images);
                    data.height = parseInt(data.height);
                    data.width = parseInt(data.width);

                    document.getElementById('loading').classList.remove('hidden');
                    document.getElementById('generate-button').disabled = true;
                    document.getElementById('results').innerHTML = '';

                    try {
                        const controller = new AbortController();
                        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5-minute timeout

                        const response = await fetch('/generate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(data),
                            signal: controller.signal
                        });

                        clearTimeout(timeoutId);

                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }

                        const result = await response.json();
                        const resultsDiv = document.getElementById('results');
                        resultsDiv.innerHTML = '';
                        result.images.forEach((image, index) => {
                            const imgContainer = document.createElement('div');
                            imgContainer.className = 'relative';

                            const img = document.createElement('img');
                            img.src = `data:image/png;base64,${image}`;
                            img.alt = `Generated Image ${index + 1}`;
                            img.className = 'w-full h-auto rounded-lg shadow-md cursor-pointer';
                            img.onclick = () => showFullscreen(img.src);

                            const saveButton = document.createElement('button');
                            saveButton.textContent = 'Save';
                            saveButton.className = 'absolute bottom-2 right-2 bg-blue-500 text-white px-2 py-1 rounded';
                            saveButton.onclick = (e) => {
                                e.stopPropagation();
                                saveImage(image, `generated_image_${index + 1}.png`);
                            };

                            imgContainer.appendChild(img);
                            imgContainer.appendChild(saveButton);
                            resultsDiv.appendChild(imgContainer);
                        });
                    } catch (error) {
                        console.error('Error:', error);
                        alert(`An error occurred: ${error.message}`);
                    } finally {
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('generate-button').disabled = false;
                    }
                });

                function showFullscreen(src) {
                    const fullscreenView = document.getElementById('fullscreen-view');
                    const fullscreenImage = document.getElementById('fullscreen-image');
                    fullscreenImage.src = src;
                    fullscreenView.classList.remove('hidden');
                    fullscreenView.onclick = () => fullscreenView.classList.add('hidden');
                }

                function saveImage(base64Data, fileName) {
                    const link = document.createElement('a');
                    link.href = `data:image/png;base64,${base64Data}`;
                    link.download = fileName;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            </script>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
