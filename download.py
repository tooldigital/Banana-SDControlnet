# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import pipeline
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

def download_model():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet,torch_dtype=torch.float16)
    depth_estimator = pipeline('depth-estimation')

if __name__ == "__main__":
    download_model()
