import os
import subprocess
import uuid
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from threading import Thread
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import glob
import re
import json
import requests
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables
jobs = defaultdict(dict)

# Configuration
DEFAULT_CONFIG = {
    "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "photo of a person",
    "push_to_hub": True,
    "hf_token": "your_huggingface_token_here",
    "hf_username": "your_huggingface_username_here",
    "learning_rate": 1e-4,
    "num_steps": 4,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "resolution": 1024,
    "use_8bit_adam": True,
    "use_xformers": True,
    "mixed_precision": "fp16",
    "train_text_encoder": False,
    "disable_gradient_checkpointing": False,
    "callback_url": "http://example.com/callback",
}

# Use AWS params from env vars
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

# Add this near the top of your file, after the imports
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    raise EnvironmentError("AWS credentials are not properly set in environment variables.")

def upload_safetensor_to_s3(job_id, project_name, training_args):
    try:
        # Create a boto3 client
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3 = session.client('s3')

        # Find the generated .safetensor file
        safetensor_files = glob.glob(f"{project_name}/*.safetensors")
        if not safetensor_files:
            logging.error(f"No .safetensor file found for job {job_id}")
            return False

        safetensor_file = safetensor_files[0]
        safetensor_name = os.path.basename(safetensor_file)

        # Create the folder structure in the S3 bucket if it doesn't exist
        s3.put_object(Bucket="myloras", Key=f"{project_name}/")

        # Upload the safetensor file to S3 with the job_id as the filename
        s3_key = f"{project_name}/{job_id}.safetensors"
        s3.upload_file(safetensor_file, "myloras", s3_key)

        # Create metadata JSON
        metadata = {
            "job_id": job_id,
            "project_name": project_name,
            "person_name": training_args["person_name"],
            "training_params": {
                "model_name": training_args["model_name"],
                "prompt": training_args["prompt"],
                "learning_rate": training_args["learning_rate"],
                "num_steps": training_args["num_steps"],
                "batch_size": training_args["batch_size"],
                "gradient_accumulation": training_args["gradient_accumulation"],
                "resolution": training_args["resolution"],
                "use_8bit_adam": training_args["use_8bit_adam"],
                "use_xformers": training_args["use_xformers"],
                "mixed_precision": training_args["mixed_precision"],
                "train_text_encoder": training_args["train_text_encoder"],
                "disable_gradient_checkpointing": training_args["disable_gradient_checkpointing"]
            },
            "timestamp": datetime.now().isoformat()
        }

        # Upload metadata JSON to S3
        metadata_json = json.dumps(metadata, indent=2)
        metadata_key = f"{project_name}/{job_id}_metadata.json"
        s3.put_object(Body=metadata_json, Bucket="myloras", Key=metadata_key)

        logging.info(f"Uploaded {s3_key} and metadata to S3 bucket 'myloras' for job {job_id}")
        return True

    except ClientError as e:
        logging.error(f"Error uploading files to S3: {e}")
        return False
    except NoCredentialsError:
        logging.error("AWS credentials not found or invalid")
        return False

def download_s3_images(bucket_name, s3_folder, local_dir=None):
    try:
        # Create a boto3 client
        session = boto3.Session(
           aws_access_key_id=AWS_ACCESS_KEY_ID,
           aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
           region_name=AWS_REGION  )
        s3 = session.client('s3')

        # Create a random folder name if not provided
        if local_dir is None:
            local_dir = str(uuid.uuid4())

        # Create the local directory
        os.makedirs(local_dir, exist_ok=True)

        # List objects within the S3 folder
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        if 'Contents' not in result:
            logging.info(f"No objects found in {s3_folder}")
            return

        # Download each object
        for obj in result['Contents']:
            # Get the file path
            s3_file = obj['Key']
            
            # Skip if it's a folder
            if s3_file.endswith('/'):
                continue
            
            # Remove the folder name from the file path
            local_file = s3_file.replace(s3_folder, '', 1).lstrip('/')
            local_file_path = os.path.join(local_dir, local_file)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            logging.info(f"Downloading {s3_file} to {local_file_path}")
            s3.download_file(bucket_name, s3_file, local_file_path)

        logging.info(f"All files downloaded to {local_dir}")
        return True

    except ClientError as e:
        logging.error(f"Error downloading files: {e}")
        return False
    except NoCredentialsError:
        logging.error("AWS credentials not found or invalid")
        return False

def call_callback_endpoint(job_id, project_name, s3_bucket, s3_folder, person_name, status, datetime, reason=None):
    callback_url = DEFAULT_CONFIG["callback_url"]
    payload = {
        "job_id": job_id,
        "project_name": project_name,
        "s3_bucket": s3_bucket,
        "s3_folder": s3_folder,
        "person_name": person_name,
        "status": status,
        "datetime": datetime,
        "reason": reason
    }
    try:
        response = requests.post(callback_url, json=payload)
        response.raise_for_status()
        logging.info(f"Callback sent successfully for job {job_id}")
    except requests.RequestException as e:
        logging.error(f"Failed to send callback for job {job_id}: {str(e)}")

def train_lora(job_id, args):
    logging.info(f"Starting training for job {job_id}")
    jobs[job_id]["status"] = "DOWNLOADING"
    jobs[job_id]["message"] = "Downloading images from S3"
    jobs[job_id]["start_time"] = datetime.now()
    
    # Download images from S3
    local_image_folder = f"images_{job_id}"
    s3_download_success = download_s3_images(args["s3_bucket"], args["s3_folder"], local_image_folder)
    
    if not s3_download_success:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["message"] = "Failed to download images from S3"
        return
    
    jobs[job_id]["status"] = "TRAINING"
    jobs[job_id]["message"] = "Training in progress"
    
    cmd = [
        "autotrain", "dreambooth",
        "--model", args["model_name"],
        "--project-name", args["project_name"],
        "--image-path", local_image_folder,  # Use the downloaded images folder
        "--prompt", args["prompt"],
        "--resolution", str(args["resolution"]),
        "--batch-size", str(args["batch_size"]),
        "--num-steps", str(args["num_steps"]),
        "--gradient-accumulation", str(args["gradient_accumulation"]),
        "--lr", str(args["learning_rate"]),
        "--mixed-precision", args["mixed_precision"],
        "--username", args["hf_username"]
    ]
    
    if args["use_xformers"]:
        cmd.append("--xformers")
    if args["train_text_encoder"]:
        cmd.append("--train-text-encoder")
    if args["use_8bit_adam"]:
        cmd.append("--use-8bit-adam")
    if args["disable_gradient_checkpointing"]:
        cmd.append("--disable_gradient-checkpointing")
    if args["push_to_hub"]:
        cmd.extend(["--push-to-hub", "--token", args["hf_token"]])
    
    logging.debug(f"Command for job {job_id}: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        jobs[job_id]["process"] = process
        jobs[job_id]["steps_completed"] = 0
        
        # Real-time logging and output capturing
        output = []
        for line in iter(process.stdout.readline, ''):
            logging.info(f"Job {job_id} stdout: {line.strip()}")
            output.append(line)
            
            # Check if the line contains step information
            if "Step" in line:
                try:
                    current_step = int(line.split("Step")[1].split("/")[0])
                    jobs[job_id]["steps_completed"] = current_step
                except ValueError:
                    pass
        
        process.wait()
        
        if process.returncode == 0:
            # Upload the generated .safetensor file and metadata to S3
            upload_success = upload_safetensor_to_s3(job_id, args["project_name"], args)
            if upload_success:
                jobs[job_id]["status"] = "COMPLETED"
                jobs[job_id]["message"] = "Training completed successfully. Safetensor file and metadata uploaded to S3."
                logging.info(f"Job {job_id} completed successfully and safetensor with metadata uploaded")
                call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                                       args["person_name"], "COMPLETED", datetime.now().isoformat())
            else:
                jobs[job_id]["status"] = "FAILED"
                jobs[job_id]["message"] = "Training completed, but failed to upload safetensor file and metadata to S3."
                logging.error(f"Job {job_id} training completed, but failed to upload safetensor and metadata")
                call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                                       args["person_name"], "FAILED", datetime.now().isoformat(), 
                                       "Failed to upload safetensor file and metadata to S3")
        else:
            jobs[job_id]["status"] = "FAILED"
            error_message = "\n".join(output[-10:])  # Capture last 10 lines of output
            jobs[job_id]["message"] = f"Training failed with return code {process.returncode}. Last output:\n{error_message}"
            logging.error(f"Job {job_id} failed with return code {process.returncode}")
            call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                                   args["person_name"], "FAILED", datetime.now().isoformat(), 
                                   f"Training failed with return code {process.returncode}")
    except Exception as e:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["message"] = f"An error occurred: {str(e)}"
        logging.exception(f"An error occurred in job {job_id}")
        call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                               args["person_name"], "FAILED", datetime.now().isoformat(), str(e))
    finally:
        jobs[job_id]["process"] = None
        # Clean up the local image folder
        if os.path.exists(local_image_folder):
            shutil.rmtree(local_image_folder)
        logging.info(f"Cleaned up local image folder for job {job_id}")

        # Clean up the project folder
        project_folder = args["project_name"]
        if os.path.exists(project_folder):
            shutil.rmtree(project_folder)
        logging.info(f"Cleaned up project folder for job {job_id}")

@app.route('/')
def index():
    return "Alive"

@app.route('/train', methods=['POST'])
def start_training():
    args = request.json
    required_args = ["job_id", "project_name", "s3_bucket", "s3_folder", "person_name"]
    
    for arg in required_args:
        if arg not in args:
            logging.warning(f"Missing required argument: {arg}")
            return jsonify({"error": f"Missing required argument: {arg}"}), 400
    
    # Validate project_name
    if not re.match(r'^[a-zA-Z0-9_-]{3,63}$', args["project_name"]):
        return jsonify({"error": "Invalid project_name. Use 3-63 alphanumeric characters, underscores, or hyphens."}), 400

    # Validate s3_bucket
    if not re.match(r'^[a-z0-9.-]{3,63}$', args["s3_bucket"]):
        return jsonify({"error": "Invalid s3_bucket name. Use 3-63 lowercase alphanumeric characters, dots, or hyphens."}), 400

    # Validate s3_folder
    if not re.match(r'^[a-zA-Z0-9_/-]{1,1024}$', args["s3_folder"]):
        return jsonify({"error": "Invalid s3_folder. Use alphanumeric characters, underscores, hyphens, or forward slashes."}), 400


    # Merge default config with provided args
    training_args = DEFAULT_CONFIG.copy()
    training_args.update(args)
    
    # Update prompt with person_name
    training_args["prompt"] = f"photo of {args['person_name']}"
    logging.info(f"Training args: {training_args}")

    job_id = args["job_id"]
    
    # Check if there's already a job running
    if is_server_busy():
        logging.warning(f"Attempted to start job {job_id} while another job is running")
        return jsonify({"error": "Another job is already running. Please wait for it to complete."}), 409
    
    # Check if the job already exists
    if job_id in jobs:
        logging.warning(f"Attempted to start existing job {job_id}")
        return jsonify({"error": f"Job {job_id} already exists"}), 409
    
    logging.info(f"Starting new job with ID: {job_id}")
    
    # Initialize job status
    jobs[job_id] = {
        "status": "INITIALIZING",
        "message": "Job is being set up",
        "steps_completed": 0,
        "num_steps": training_args["num_steps"]
    }
    
    # Check if 'images/' directory exists and is not empty
    if not os.path.exists('images/') or not os.listdir('images/'):
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["message"] = "'images/' directory not found or empty. Please add training images."
        logging.error(jobs[job_id]["message"])
        return jsonify({"error": jobs[job_id]["message"]}), 400
    
    # Clean up any existing project folder with the same name
    if os.path.exists(training_args["project_name"]):
        shutil.rmtree(training_args["project_name"])
        logging.info(f"Cleaned up existing project folder: {training_args['project_name']}")

    thread = Thread(target=train_lora, args=(job_id, training_args))
    thread.start()
    
    logging.info(f"Training thread started for job {job_id}")
    return jsonify({"message": "Training started", "job_id": job_id}), 200

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        logging.warning(f"Attempted to get status for non-existent job {job_id}")
        return jsonify({"error": "Job not found"}), 404
    
    status_info = {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "message": jobs[job_id].get("message", ""),
        "steps_completed": jobs[job_id].get("steps_completed", 0),
        "total_steps": jobs[job_id].get("num_steps", 0)
    }
    
    if "start_time" in jobs[job_id]:
        elapsed_time = datetime.now() - jobs[job_id]["start_time"]
        status_info["elapsed_time"] = str(elapsed_time)
    
    logging.info(f"Status for job {job_id}: {status_info}")
    return jsonify(status_info)

@app.route('/stop/<job_id>', methods=['POST'])
def stop_training(job_id):
    if job_id not in jobs:
        logging.warning(f"Attempted to stop non-existent job {job_id}")
        return jsonify({"error": "Job not found"}), 404
    
    if jobs[job_id]["status"] != "BUSY" or jobs[job_id]["process"] is None:
        logging.info(f"No training in progress for job {job_id}")
        return jsonify({"message": "No training in progress for this job"}), 400
    
    jobs[job_id]["process"].terminate()
    jobs[job_id]["process"] = None
    jobs[job_id]["status"] = "STOPPED"
    jobs[job_id]["message"] = "Training stopped by user"
    
    logging.info(f"Training for job {job_id} stopped by user")
    return jsonify({"message": f"Training for job {job_id} stopped"}), 200

@app.route('/jobs', methods=['GET'])
def list_jobs():
    job_list = {job_id: {
        "status": job_info["status"],
        "message": job_info.get("message", "")
    } for job_id, job_info in jobs.items()}
    logging.info(f"Current job list: {job_list}")
    return jsonify(job_list)

def is_server_busy():
    return any(job["status"] == "BUSY" for job in jobs.values())

@app.route('/busy', methods=['GET'])
def check_server_busy():
    busy = is_server_busy()
    logging.info(f"Server busy status: {busy}")
    return jsonify({"busy": busy})

def check_job_status():
    for job_id, job_info in jobs.items():
        if job_info["status"] == "BUSY" and (job_info["process"] is None or job_info["process"].poll() is not None):
            job_info["status"] = "FAILED"
            job_info["message"] = "Process terminated unexpectedly"
            logging.error(f"Job {job_id} process terminated unexpectedly")

if __name__ == '__main__':
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=check_job_status, trigger="interval", seconds=60)
    scheduler.start()
    app.run(host='0.0.0.0', port=5656, debug=True)