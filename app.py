from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
import os
import uuid
import json
import base64
import torch
from openai import OpenAI
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.abspath("image_data")
TEXTURES_DIR = "textures"
MODEL_DIR = "model"
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Global dictionary to store metadata
image_data = {}

# Helper function to save image and create directory structure
def save_image(image):
    # Generate unique ID and create directory for image
    img_id = str(uuid.uuid4())
    img_dir = os.path.join(BASE_DIR, img_id)
    os.makedirs(os.path.join(img_dir, TEXTURES_DIR), exist_ok=True)
    os.makedirs(os.path.join(img_dir, MODEL_DIR), exist_ok=True)
    
    # Convert and save image as JPEG
    img_path = os.path.join(img_dir, "image.jpeg")
    image.save(img_path, format="JPEG")
    
    # Initialize metadata
    image_data[img_id] = {
        "status": "processing",
        "Indoor": None,  # to be filled by LLM
        "Objects": [],
    }
    return img_id, img_path

# Route to upload an image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Process image upload
    image_file = request.files['image']
    image = Image.open(image_file)
    img_id, img_path = save_image(image)

    # Encode the image for LLM processing
    encoded_image = encode_image(img_path)
    
    # Send image to LLM and update metadata
    llm_response = send_to_llm(encoded_image)
    if llm_response:
        image_data[img_id].update(llm_response)
        image_data[img_id]["status"] = "complete"
    else:
        image_data[img_id]["status"] = "failed"

    return jsonify({"message": "Image uploaded successfully", "img_id": img_id})

# Helper function to encode an image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to send image to LLM and retrieve JSON
def send_to_llm(encoded_image):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate JSON in this rough format for images:\n\n"
                    "{\"status\": (processing, done, failed), Indoor:Bool, Objects: [list of this format "
                    "{\"type\": (table, chair , door), count: int, scatter: bool}], "
                    "\"Wall-texture\": str, \"Floor-texture\": str, \"Ceiling-texture\": str}"
                )
            },
            {
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            }
        ],
        temperature=1,
        max_tokens=500
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Error parsing LLM response:", e)
        return None

# 1. Route to list all IDs
@app.route('/list', methods=['GET'])
def list_ids():
    return jsonify(list(image_data.keys()))

# 2. Route to return JSON info for an image
@app.route('/info', methods=['GET'])
def get_info():
    img_id = request.args.get('img_id')
    if img_id not in image_data:
        return jsonify({"error": "Image ID not found"}), 404
    return jsonify(image_data[img_id])

# Route to get a file by relative path
@app.route('/file', methods=['GET'])
def get_file():
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"error": "File path not provided"}), 400

    abs_path = os.path.join(BASE_DIR, file_path)
    if not os.path.isfile(abs_path):
        abort(404, description="File not found")

    return send_file(abs_path, as_attachment=True)

# Load texture generation model
model_id = "dream-textures/texture-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Function to generate textures based on prompts
def generate_texture(prompt, save_path):
    if prompt:
        generated_image = pipe(prompt).images[0]
        generated_image.save(save_path)

# Function to update image_data with generated textures
@app.route('/generate_textures', methods=['POST'])
def generate_textures():
    img_id = request.json.get("img_id")
    if img_id not in image_data or image_data[img_id]["status"] != "complete":
        return jsonify({"error": "Invalid image ID or image processing incomplete"}), 400

    # Get texture prompts
    wall_prompt = image_data[img_id].get("Wall-texture")
    floor_prompt = image_data[img_id].get("Floor-texture")
    ceiling_prompt = image_data[img_id].get("Ceiling-texture")
    
    # Generate textures
    img_dir = os.path.join(BASE_DIR, img_id, TEXTURES_DIR)
    generate_texture(wall_prompt, os.path.join(img_dir, "wall.png"))
    generate_texture(floor_prompt, os.path.join(img_dir, "floor.png"))
    generate_texture(ceiling_prompt, os.path.join(img_dir, "ceiling.png"))

    return jsonify({"message": "Textures generated successfully"})

if __name__ == '__main__':
    app.run(debug=True)
