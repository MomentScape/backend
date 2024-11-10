# docker run -it --platform=linux/amd64 --gpus all -e CUDA_HOME="/usr/local/cuda/" -p 5100:5100 img2threed

import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile

from huggingface_hub import hf_hub_download



def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(50.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

import shutil

def find_cuda():
    # Check if CUDA_HOME or CUDA_PATH environment variables are set
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

    if cuda_home and os.path.exists(cuda_home):
        return cuda_home

    # Search for the nvcc executable in the system's PATH
    nvcc_path = shutil.which('nvcc')

    if nvcc_path:
        # Remove the 'bin/nvcc' part to get the CUDA installation path
        cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
        return cuda_path

    return None

cuda_path = find_cuda()

if cuda_path:
    print(f"CUDA installation found at: {cuda_path}")
else:
    print("CUDA installation not found")

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model")
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)

print('Loading Finished!')




def preprocess(input_image, do_remove_background):

    rembg_session = rembg.new_session() if do_remove_background else None

    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def generate_mvs(input_image, sample_steps, sample_seed):

    seed_everything(sample_seed)
    
    # sampling
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())

    return z123_image, show_image


def make3d(images, out_path):

    global model
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, use_renderer=False)
    model = model.eval()

    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(batch_size=1, radius=2.5, is_flexicubes=IS_FLEXICUBES).to(device)

    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )
        
        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        
        save_glb(vertices, faces, vertex_colors, out_path)
        # save_obj(vertices, faces, vertex_colors, mesh_fpath)

    # vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
    # save_obj_with_mtl(
    #             vertices.data.cpu().numpy(),
    #             uvs.data.cpu().numpy(),
    #             faces.data.cpu().numpy(),
    #             mesh_tex_idx.data.cpu().numpy(),
    #             tex_map.permute(1, 2, 0).data.cpu().numpy(),
    #             out_path,
    #         )



    return out_path


# 📝 **Citation**

# If you find our work useful for your research or applications, please cite using this bibtex:
# ```bibtex
# @article{xu2024instantmesh,
#   title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
#   author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
#   journal={arXiv preprint arXiv:2404.07191},
#   year={2024}
# }



from flask import Flask, request, jsonify, send_file
import uuid

# Import necessary functions and models
# Assuming your existing functions and imports are in a module or accessible here.
# Adjust imports as needed based on your structure.

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/img2threeD', methods=['POST'])
def img2threeD():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Check if the user selected a file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filename = str(uuid.uuid4())
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    
    # Parameters (these could be taken from request args if you want to make it configurable)
    do_remove_background = int(request.form.get('do_remove_background', 1))
    sample_steps = int(request.form.get('sample_steps', 75))
    sample_seed = int(request.form.get('sample_seed', 42))

    try:
        # Process the image and generate the 3D model
        processed_image = preprocess(Image.open(image_path), do_remove_background)
        mv_images, mv_show_images = generate_mvs(processed_image, sample_steps, sample_seed)

        # Generate a temporary file path for the 3D output
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            output_path = temp_file.name

        # Create the 3D model
        make3d(mv_images, output_path)

        # Send the file as a response
        return send_file(output_path, as_attachment =True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)

