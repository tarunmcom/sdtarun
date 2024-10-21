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
from PIL import Image

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

# Modify the jobs dictionary to include more information
jobs = {}
jobs_lock = threading.Lock()

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
        # Original image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        
        # Create thumbnail
        thumb_size = (200, 200)  # You can adjust this size as needed
        thumbnail = image.copy()
        thumbnail.thumbnail(thumb_size)
        thumb_buffered = BytesIO()
        thumbnail.save(thumb_buffered, format="PNG")
        
        # Create the full S3 paths
        s3_path = f"{s3_folder_path}/{filename}"
        thumb_filename = f"{os.path.splitext(filename)[0]}_thumb.png"
        thumb_s3_path = f"{s3_folder_path}/{thumb_filename}"
        
        # Ensure the folder structure exists
        folder_parts = s3_folder_path.split('/')
        for i in range(1, len(folder_parts) + 1):
            folder_path = '/'.join(folder_parts[:i]) + '/'
            s3.put_object(Bucket=bucket_name, Key=folder_path, Body='')
        
        # Upload the original image
        s3.put_object(Bucket=bucket_name, Key=s3_path, Body=buffered.getvalue())
        logger.info(f"Uploaded image to S3: {s3_path}")
        
        # Upload the thumbnail
        s3.put_object(Bucket=bucket_name, Key=thumb_s3_path, Body=thumb_buffered.getvalue())
        logger.info(f"Uploaded thumbnail to S3: {thumb_s3_path}")
        
        return f"{s3_folder_path}/{filename}"  # Return the full S3 path of the original image
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
                "text": "system_instruction=\"You are a helpful Prompt Engineer. You are tasked to refine user's input prompt for image generation using stable diffusion model version SDXL.The SDXL model generates good images if only the human subject is mostly visible and there is no or minimal background. If you try to generate wide angle shot or too many subjects in the image, then SDXL gives poor response, so you should avoid this by refining the prompt. You should refine the prompts in such a way that it follows users input prompt but the main human surject must occupy the maximum space in the image and background is minimal or blurred. You should also utilize the sdxl prompt generation skills by providing shot description, camera and lenses parameters that generate these type of shots. You will receive the name of a person in encoded form and you should use that in your prompt. Example:\\n'''\\nUser: {\\\"prompt\\\":\\\"I am riding a horse\\\", \\\"name\\\":\\\"Tarun_qwer\\\"} \\nrefined prompt:- \\\"A close-up of Tarun_qwer on a horse, with focus on the detailed subject while the background is blurred.Tarun_qwer is in riding attire, and the horse is in motion, emphasizing their expressions and textures, Tarun_qwer has proper eyes\\\"\\n'''\\nNever use wide angle or long distance shots. You may use the following shots 1. close up shot 2. medium shot 3. portrait. You may also use lenses focus and parameters that generate these type of shots.\\nAlso add relevant info so that face is generated well. And generate 4-5 different prompts from one prompt.\\nAlways generate prompt in 50-60 words because above that response is not good.\\nMake sure you give output in the following format:-\\n'''\\n {\\\"prompts\\\":[\\\"<PROMPT>\\\", \\\"<PROMPT>\\\",....]}\\n'''\\nYou must always generate prompts so that human subject occupies the maximum area in the image and more detailing on human face. You must add relevant face expression of the human subject more, so that face detailing is good. Image Generation models are bad in generating hands and fingers, so you should cleverly prompt to hide hands and finger behind any relevant object or in pocket or behind back etc. \","            }
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

def update_job_status(job_id: str, status: str, error_message=None):
    with jobs_lock:  # Ensure thread safety
        job = get_job_by_id(job_id)
        if job is None:
            job = {'job_id': job_id, 'status': status}
        else:
            job['status'] = status
        if error_message:
            job['error_message'] = error_message
        save_job(job)
    
    # Update external status if endpoint is set
    status_update_endpoint = os.getenv('STATUS_UPDATE_ENDPOINT')
    if status_update_endpoint:
        try:
            response = requests.post(
                status_update_endpoint,
                json={'job_id': job_id, 'status': status},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to update status at {status_update_endpoint}: {str(e)}")
    else:
        logger.info("STATUS_UPDATE_ENDPOINT not set, skipping external status update")

    # Update global status
    update_global_status(job_id, status)

def update_global_status(job_id: str, status: str):
    global current_job_id, global_status
    with jobs_lock:  # Ensure thread safety
        if status in ['completed', 'failed']:
            current_job_id = None
            global_status = 'IDLE'
        else:
            current_job_id = job_id
            global_status = 'BUSY'

# In-memory job storage for demonstration purposes
jobs = {}

def get_job_by_id(job_id):
    return jobs.get(job_id)

def save_job(job):
    if 'job_id' not in job:
        raise ValueError("Job must have a 'job_id' key")
    jobs[job['job_id']] = job

def process_image_generation(job_id, data):
    try:
        with jobs_lock:
            jobs[job_id] = {'job_id': job_id, 'status': 'initializing'}
        update_job_status(job_id, 'initializing')
        
        lora_path = data.get('lora_path')
        crude_prompt = data.get('crude_prompt')
        negative_prompt = data.get('negative_prompt', '')
        num_steps = data.get('num_steps', 40)
        num_images = data.get('num_images', 1)
        height = data.get('height', 1024)
        width = data.get('width', 1024)
        output_s3_folder_path = data.get('output_s3_folder_path')

        update_job_status(job_id, 'preparing_lora')
        s3_bucket = 'myloras'  # Replace with your actual S3 bucket name
        local_lora_path = get_local_lora_path(lora_path)
        download_lora_from_s3(s3_bucket, lora_path, local_lora_path)

        update_job_status(job_id, 'getting_lora_metadata')
        lora_metadata_path = lora_path.rsplit('.', 1)[0] + '_metadata.json'
        lora_user_name = get_lora_metadata(s3_bucket, lora_metadata_path)

        update_job_status(job_id, 'refining_prompt')
        llm_response = generate_llm_response(crude_prompt, lora_user_name)
        response_json = convert_to_json(llm_response)
        refined_prompts = make_prompt_list(num_images, response_json)

        update_job_status(job_id, 'generating_seeds')
        seeds = [random.randint(0, 4294967295) for _ in range(num_images)]
        generators = [torch.Generator("cuda").manual_seed(seed) for seed in seeds]

        update_job_status(job_id, 'loading_lora_weights')
        pipe.load_lora_weights(local_lora_path)

        update_job_status(job_id, 'generating_images')
        batch_images = pipe(
            prompt=refined_prompts,
            negative_prompt=[negative_prompt] * num_images,
            generator=generators,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            num_images_per_prompt=1
        ).images

        update_job_status(job_id, 'uploading_images')
        s3_image_paths = []
        generated_images_s3_bucket = 'genimagesdxl'
        
        # Parse the output_s3_folder_path
        user_id, lora_id, predict_job_id = data['output_s3_folder_path'].split('/')
        
        for i, image in enumerate(batch_images):
            filename = f"{i}_{uuid.uuid4()}.png"
            s3_folder_path = f"{user_id}/{lora_id}/{predict_job_id}"
            s3_path = upload_image_to_s3(generated_images_s3_bucket, image, s3_folder_path, filename)
            # Construct the correct S3 path for the response
            correct_s3_path = f"{s3_folder_path}/{filename}"
            s3_image_paths.append(correct_s3_path)

        update_job_status(job_id, 'unloading_lora_weights')
        pipe.unload_lora_weights()

        update_job_status(job_id, 'completed')
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['result'] = {
                    "job_id": job_id,
                    "image_paths": s3_image_paths,
                    "seeds": seeds,
                    "refined_prompts": refined_prompts,
                    "crude_prompt": crude_prompt  # Add the original crude prompt
                }
            else:
                logger.warning(f"Job {job_id} not found in jobs dictionary when trying to update result")

    except Exception as e:
        logger.error(f"Error in process_image_generation: {str(e)}", exc_info=True)
        update_job_status(job_id, 'failed', str(e))

@app.route('/generate_images', methods=['POST'])
def generate_images():
    with jobs_lock:
        busy_jobs = [job for job in jobs.values() if job.get('status') not in ['completed', 'failed']]
        if busy_jobs:
            return jsonify({"error": "Server is busy processing another request"}), 503

    logger.info("Received request to generate images")
    data = request.json
    logger.info(f"Request data: {data}")

    # Use the job_id provided in the payload
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id is required in the request payload"}), 400

    update_job_status(job_id, 'initializing')

    thread = threading.Thread(target=process_image_generation, args=(job_id, data))
    thread.start()

    return jsonify({'job_id': job_id}), 202

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = get_job_by_id(job_id)
    if job:
        response = {'job_id': job_id, 'status': job['status']}
        
        if job['status'] == 'failed':
            response['error_message'] = job.get('error_message', 'Unknown error')
        
        elif job['status'] == 'completed':
            result = job.get('result', {})
            response.update({
                'image_paths': result.get('image_paths', []),
                'seeds': result.get('seeds', []),
                'refined_prompts': result.get('refined_prompts', []),
                'crude_prompt': result.get('crude_prompt', '')  # Add the crude prompt to the response
            })
        
        return jsonify(response)
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'current_job_id': current_job_id, 'status': global_status})

@app.route('/busy', methods=['GET'])
def get_busy_status():
    with jobs_lock:
        return jsonify({"busy": any(job.get('status') not in ['completed', 'failed'] for job in jobs.values())})

@app.route('/')
def index():
    return jsonify({"message": "alive", "server_type":"Predict"})  # Updated to return JSON response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
