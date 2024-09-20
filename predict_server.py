from flask import Flask, request, jsonify
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
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
import re
import json
import uuid
import threading
import os.path
import requests

# Use AWS params from env vars
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

# Add this near the top of your file, after the imports
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    raise EnvironmentError("AWS credentials are not properly set in environment variables.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

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

# Add these global variables
current_lora_weights = None
lora_cache = {}

# Initialize S3 client
session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
s3 = session.client('s3')

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Add these global variables
current_job_id = None
server_status = "IDLE"
job_status = {}
status_lock = threading.Lock()

# Add this near the top of your file with other environment variables
STATUS_UPDATE_ENDPOINT = os.environ.get('STATUS_UPDATE_ENDPOINT')

def get_local_lora_path(lora_path):
    folder_name, lora_name = os.path.split(lora_path)
    return f"./lora_cache/{folder_name}_{lora_name}"

def download_lora_from_s3(bucket_name, lora_path, local_path):
    if local_path in lora_cache:
        logger.info(f"Using cached LORA: {local_path}")
        return

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, lora_path, local_path)
        logger.info(f"Downloaded LORA from S3: {lora_path}, saved to {local_path}")
        lora_cache[local_path] = True
    except ClientError as e:
        logger.error(f"Error downloading LORA from S3: {str(e)}")
        raise

def upload_image_to_s3(bucket_name, image, s3_folder_path, filename):
    try:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        s3_path = f"{s3_folder_path}/{filename}"
        s3.put_object(Bucket=bucket_name, Key=s3_path, Body=buffered.getvalue())
        logger.info(f"Uploaded image to S3: {s3_path}")
        return s3_path
    except ClientError as e:
        logger.error(f"Error uploading image to S3: {str(e)}")
        raise

def make_prompt_list(num_images, response_json):
    prompts = json.loads(response_json)["prompts"]
    refined_prompts = []
    for _ in range(num_images):
        if isinstance(prompts[0], dict):
            # Handle the case where prompts is a list of dictionaries
            refined_prompts.append(prompts[random.randint(0, len(prompts)-1)]["prompt"])
        else:
            # Handle the case where prompts is a list of strings
            refined_prompts.append(prompts[random.randint(0, len(prompts)-1)])
    return refined_prompts

def convert_to_json(input_text):
    # Check if the input is already valid JSON
    try:
        data = json.loads(input_text)
        if isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
            return input_text  # Return as-is if it's already in the correct format
    except json.JSONDecodeError:
        pass  # Not valid JSON, continue with existing logic

    # Remove any surrounding backticks if present
    input_text = input_text.strip('`')

    try:
        data = json.loads(input_text)
        if isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
            return json.dumps(data)
    except json.JSONDecodeError:
        pass  # Not valid JSON, raise error

    raise ValueError(
        "Invalid input format. Expected format:\n"
        '{"prompts": ["prompt1", "prompt2", ...]}\n'
        'or valid JSON with "prompts" key containing a list.'
    )

def generate_llm_response(crude_prompt, lora_user_name):
    crude_prompt = f"prompt: {crude_prompt}, name: {lora_user_name}"
    logging.info(f"crude_prompt: {crude_prompt}")
    response = client.chat.completions.create(
        model="gpt-4",
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
        presence_penalty=0
    )
    res = response.choices[0].message.content
    return res

def get_lora_metadata(bucket_name, lora_path):
    local_metadata_path = get_local_lora_path(lora_path).rsplit('.', 1)[0] + '.json'
    
    if os.path.exists(local_metadata_path):
        with open(local_metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('person_name', 'anonymous')

    metadata_path = lora_path.rsplit('.', 1)[0] + '.json'
    try:
        logger.info(f"Downloading LORA metadata from S3: {metadata_path} , bucket: {bucket_name}")
        response = s3.get_object(Bucket=bucket_name, Key=metadata_path)
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        # Save metadata locally
        os.makedirs(os.path.dirname(local_metadata_path), exist_ok=True)
        with open(local_metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return metadata.get('person_name', 'anonymous')
    except ClientError as e:
        logger.error(f"Error reading LORA metadata from S3: {str(e)}")
        return 'anonymous'

def update_job_status(job_id, status, error_message=None):
    if not STATUS_UPDATE_ENDPOINT:
        logger.warning("STATUS_UPDATE_ENDPOINT not set. Skipping status update.")
        return

    payload = {
        "job_id": job_id,
        "status": status
    }
    if error_message:
        payload["error_message"] = error_message

    try:
        response = requests.post(STATUS_UPDATE_ENDPOINT, json=payload)
        response.raise_for_status()
        logger.info(f"Status update sent for job {job_id}: {status}")
    except requests.RequestException as e:
        logger.error(f"Failed to send status update for job {job_id}: {str(e)}")

jobs = {}

def process_image_generation(job_id, data):
    try:
        jobs[job_id]['status'] = 'initializing'
        
        lora_path = data.get('lora_path')
        crude_prompt = data.get('crude_prompt')
        negative_prompt = data.get('negative_prompt', '')
        num_steps = data.get('num_steps', 40)
        num_images = data.get('num_images', 1)
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        output_s3_folder_path = data.get('output_s3_folder_path')

        jobs[job_id]['status'] = 'preparing_lora'
        s3_bucket = 'myloras'  # Replace with your actual S3 bucket name
        local_lora_path = get_local_lora_path(lora_path)
        download_lora_from_s3(s3_bucket, lora_path, local_lora_path)

        jobs[job_id]['status'] = 'getting_lora_metadata'
        lora_metadata_path = lora_path.rsplit('.', 1)[0] + '_metadata.json'
        lora_user_name = get_lora_metadata(s3_bucket, lora_metadata_path)

        jobs[job_id]['status'] = 'refining_prompt'
        llm_response = generate_llm_response(crude_prompt, lora_user_name)
        response_json = convert_to_json(llm_response)
        refined_prompts = make_prompt_list(num_images, response_json)

        jobs[job_id]['status'] = 'generating_seeds'
        seeds = [random.randint(0, 4294967295) for _ in range(num_images)]
        generators = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]

        jobs[job_id]['status'] = 'loading_lora_weights'
        pipe.load_lora_weights(local_lora_path)

        jobs[job_id]['status'] = 'generating_images'
        batch_images = pipe(
            prompt=refined_prompts,
            negative_prompt=[negative_prompt] * num_images,
            generator=generators,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            num_images_per_prompt=1
        ).images

        jobs[job_id]['status'] = 'uploading_images'
        s3_image_paths = []
        generated_images_s3_bucket = 'genimagesdxl'
        for i, image in enumerate(batch_images):
            filename = f"{job_id}_{i}_{uuid.uuid4()}.png"
            s3_path = upload_image_to_s3(generated_images_s3_bucket, image, output_s3_folder_path, filename)
            s3_image_paths.append(s3_path)

        jobs[job_id]['status'] = 'unloading_lora_weights'
        pipe.unload_lora_weights()

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            "job_id": job_id,
            "image_paths": s3_image_paths,
            "seeds": seeds,
            "refined_prompts": refined_prompts
        }

    except Exception as e:
        logger.error(f"Error in process_image_generation: {str(e)}", exc_info=True)
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

    finally:
        update_job_status(job_id, jobs[job_id]['status'], jobs[job_id].get('error'))

@app.route('/generate_images', methods=['POST'])
def generate_images():
    global server_status

    with status_lock:
        if server_status == "BUSY":
            return jsonify({"error": "Server is busy processing another request"}), 503

        server_status = "BUSY"

    logger.info("Received request to generate images")
    data = request.json
    logger.info(f"Request data: {data}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'initializing'}

    thread = threading.Thread(target=process_image_generation, args=(job_id, data))
    thread.start()

    return jsonify({'job_id': job_id}), 202

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    response = {'status': job['status']}

    if job['status'] == 'completed':
        response['result'] = job['result']
    elif job['status'] == 'failed':
        response['error'] = job['error']

    return jsonify(response)

@app.route('/status', methods=['GET'])
def get_status():
    with status_lock:
        return jsonify({
            "status": server_status,
            "current_job_id": current_job_id
        })
    
@app.route('/')
def index():
    return jsonify({"message": "alive"})  # Updated to return JSON response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
