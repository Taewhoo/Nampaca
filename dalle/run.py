import openai
import gradio as gr
from PIL import Image
from io import BytesIO
import base64
import requests
import random

mykey = "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"

def toRandIm(outputs_list):  # when n > 1
    img_store = []

    for output in outputs_list:
        img_url = output["url"]
        img_store.append(img_url)
    return random.choice(img_store)
###########################################################################################
# 1. create: t -> i
def tti_create(text):
    openai.api_key = mykey
    response = openai.Image.create(
    prompt = f"{text}",
    n=1,
    size="256x256")
    outputs = response['data'] 
    
    result = toRandIm(outputs)

    return result

# 2. ⭐️ 먼가 더 편리할 필요가 있음. edit: edit and extend an image by uploading a mask
def tti_edit(image_to_editP, maskP, text): # must both be square PNG images less than 4MB in size
    openai.api_key = mykey
    response = openai.Image.create_edit(
    image=open(f"{image_to_editP}", "rb"),
    mask=open(f"{maskP}", "rb"), # The non-transparent areas of the mask are not used when generating the output, so they don’t necessarily need to match the original image 
    prompt=f"{text}",
    n=1,
    size="256x256")
    outputs = response['data']

    result = toRandIm(outputs)

    return result

# 3. variation: a variation of a given image.
def tti_var(image_to_varP):
    openai.api_key = mykey
    response = openai.Image.create_variation(
    image=open(f"{image_to_varP}", "rb"),
    n=1,
    size="1024x1024")
    outputs = response['data']

    result = toRandIm(outputs)

    return result
###########################################################################################

def demo(fun):
    if fun == tti_create:
        gr_input = gr.inputs.Textbox(label="Input Text")

    elif fun == tti_edit:
        gr_input = gr.inputs.Textbox(label="Input Text")

    elif fun == tti_var:
        gr_input = gr.inputs.Textbox(label="Input Text")

    gr_output = gr.outputs.Image(type="filepath")

    gr.Interface(fn=fun, inputs=gr_input, outputs=gr_output, title="Text to Image Converter").launch()

demo(tti_create)