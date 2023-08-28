#https://github.com/lucataco/banana-controlnet
#https://huggingface.co/models?pipeline_tag=depth-estimation


from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():


    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet,torch_dtype=torch.float16)
    model.to("cuda")
    #model.enable_model_cpu_offload()
    #model.enable_xformers_memory_efficient_attention()


    '''controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )'''
   
    depth_estimator = pipeline('depth-estimation')

    context = {
        "model": model,
        "depth_estimator":depth_estimator
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:

    model = context.get("model")
    depth_estimator = context.get("depth_estimator")

    prompt = request.json.get("prompt")
    negative_prompt = request.json.get("negative_prompt")
    num_inference_steps = request.json.get("steps")
    image_data = request.json.get("image_data")
    
    #create depth image
    incimage = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

    image = depth_estimator(incimage)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)


    buffered = BytesIO()
    control_image.save(buffered,format="PNG")
    depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
 
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
   

    output = model(
        prompt,
        control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps
    )

    out_image = output.images[0]
    buffered = BytesIO()
    out_image.save(buffered,format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
   
    return Response(
        json = {"image":image_base64}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
