import openai
import gradio as gr
from PIL import Image
import io
from io import BytesIO
import base64
import requests
import random

import numpy as np

mykey = ##insert openai key##

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
    size="1024x1024")
    outputs = response['data'] 
    
    result = toRandIm(outputs)

    return result

# 2. ⭐️ 먼가 더 편리할 필요가 있음. edit: edit and extend an image by uploading a mask
def tti_edit(image_to_editP, maskP, text): # must both be square PNG images less than 4MB in size
    openai.api_key = mykey
    
    img_to_edit = Image.fromarray(image_to_editP)
    img_masked = Image.fromarray(maskP)
    img_masked = img_masked.convert("RGBA")

    with io.BytesIO() as output1:
        img_to_edit.save(output1, format='PNG')
        img_to_edit_bytes = output1.getvalue() # convert array -> image obj. -> bytes obj.
        
    with io.BytesIO() as output2:
        img_masked.save(output2, format='PNG')
        img_masked_bytes = output2.getvalue()
    
    response = openai.Image.create_edit(
    image=img_to_edit_bytes,
    mask=img_masked_bytes, # The non-transparent areas of the mask are not used when generating the output, so they don’t necessarily need to match the original image 
    prompt=f"{text}",
    n=1,
    size="1024x1024"
    )
    outputs = response['data'][0]['url']
    print(outputs)

    # result = toRand Im(outputs)
    result = outputs
    return result

# 3. variation: a variation of a given image.
def tti_var(image_to_varP):
    openai.api_key = mykey
    
    img = Image.fromarray(image_to_varP)
    with io.BytesIO() as output:
        img.save(output, format='PNG')
        image_bytes = output.getvalue() # convert array -> image obj. -> bytes obj.
    
    response = openai.Image.create_variation(
    image=image_bytes,
    n=1,
    size="1024x1024")
    outputs = response['data']
    print(outputs)
    result = toRandIm(outputs)

    return result
###########################################################################################

def demo(fun):
    if fun == tti_create:
        gr_input = gr.Textbox(label="Input text")

    elif fun == tti_edit:
        gr_input = [
            gr.Image(label="Input image"),
            gr.Image(label="Input masked image"),
            gr.Textbox(label="Input text")
        ]
    elif fun == tti_var:
        gr_input = gr.Image(label="Input image")

    gr_output = gr.Image(type="filepath")

    gr.Interface(fn=fun, inputs=gr_input, outputs=gr_output, title="Text to Image Converter").launch(share=True)

demo(tti_var)
