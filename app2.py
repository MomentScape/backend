from flask import Flask, jsonify, request, send_file, abort
from PIL import Image
import os
import uuid
import json
import base64
import torch
from openai import OpenAI
from diffusers import StableDiffusionPipeline

# Load texture generation model
model_id = "dream-textures/texture-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.abspath("image_data")
TEXTURES_DIR = "textures"
MODEL_DIR = "model"
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(
    api_key="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"
)

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

import requests

# Route to upload an image
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Process image upload
    image_file = request.files["image"]
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
        
    # doimg2threeD(img_path, os.path.join(BASE_DIR, img_id, "model", "asset.glb")) 
    # Call the /img2threeD API
    url = "http://localhost:5100/img2threeD"
    files = {'image': open(img_path, 'rb')}
    params = {
        'do_remove_background': 1,  # or any other value as needed
        'sample_steps': 75,        # or customize based on your requirement
        'sample_seed': 42          # or customize based on your requirement
    }
    try:
        response = requests.post(url, files=files, data=params)
        if response.status_code == 200:
            # Save the 3D output file if needed
            output_path = os.path.join(BASE_DIR, img_id, "model", "asset.glb")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"3D model saved at {output_path}")
        else:
            print(f"3D generation failed with status code {response.status_code}: {response.json()}")
    except Exception as e:
        print(f"Error calling 3D generation API: {e}")
    
    
    

    generate_textures(img_id)

    return jsonify({"message": "Image uploaded successfully", "img_id": img_id})


# Helper function to encode an image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Helper function to send image to LLM and retrieve JSON
def send_to_llm(encoded_image):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate JSON in this rough format for images:\n\n"
                    '"Wall-texture": str prompt for the texture to be detail-oriented with colors and pattern information, "Floor-texture": str prompt for the texture to be detail-oriented with colors and pattern information, '
                    '"Ceiling-texture": str prompt for the texture to be detail-oriented with colors and pattern information}\n\n'
                    "Return only this JSON and nothing else, add no extra fields."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    }
                ],
            },
        ],
        temperature=1,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Error parsing LLM response:", e)
        return None


# 1. Route to list all IDs
@app.route("/list", methods=["GET"])
def list_ids():
    return jsonify(list(image_data.keys()))


# 2. Route to return JSON info for an image
@app.route("/info", methods=["GET"])
def get_info():
    img_id = request.args.get("img_id")
    if img_id not in image_data:
        return jsonify({"error": "Image ID not found"}), 404
    return jsonify(image_data[img_id])


# Route to get a file by relative path
@app.route("/file", methods=["GET"])
def get_file():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "File path not provided"}), 400

    abs_path = os.path.join(BASE_DIR, file_path)
    if not os.path.isfile(abs_path):
        abort(404, description="File not found")

    return send_file(abs_path, as_attachment=True)


# Function to generate textures based on prompts
def generate_texture(prompt, save_path):
    if prompt:
        generated_image = pipe(prompt).images[0]
        generated_image.save(save_path)


# Function to update image_data with generated textures
# @app.route("/generate_textures", methods=["POST"])
def generate_textures(img_id):
    # img_id = request.json.get("img_id")
    if img_id not in image_data or image_data[img_id]["status"] != "complete":
        return (
            jsonify({"error": "Invalid image ID or image processing incomplete"}),
            400,
        )

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


if __name__ == "__main__":
    app.run(debug=True)
