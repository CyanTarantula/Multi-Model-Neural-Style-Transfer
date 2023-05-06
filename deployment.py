import streamlit as st
import pandas as pd
import numpy as np
from  PIL import Image, ImageEnhance
import pickle
from streamlit import config
import style_transfer_functions
import time
import tempfile
import pathlib
import requests

st.set_page_config(page_title="Style Transfer - DLOps", page_icon="🎭")

model_vae = style_transfer_functions.VAE()

st.sidebar.title("Configurations")
st.title("Style Transfer using Different Architectures")

###### SIDEBAR  #######

st.sidebar.header("Select Model Architecture")

model_options = {
    'Variational AE': model_vae,
    'Tranformer' : None,
    'Style-GAN' : None,
    'Pics-Art API' : None,
}
selected_model = st.sidebar.radio("Models",tuple(model_options.keys()))

with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to convert your normal Images into different styles.\nUpload an Image that you want to transform, and another Image which is the Style reference. What you get as a result is the original image stylized according to the reference style Image.\n\nThis app was created by Soumik, Yash and Stuti as a part of the DLOps project for the course CSL4020: Deep Learning offered at IIT Jodhpur during Jan-May 2023.
     """)

output_img_warning = None
output_img_error = None

st.subheader('Upload your Image and Style Reference')
content_file = st.file_uploader("Image to Style", type=['jpg','png','jpeg'])
style_file = st.file_uploader("Style Reference Image", type=['jpg','png','jpeg'])

def upload_file(file):
    url = "https://api.imgur.com/3/image"

    payload={'image': file.read()}

    files=[]

    headers = {
        'Authorization': 'Client-ID 52d1141cb13f609'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return response.json()

def delete_file(file_response):
    url = f"https://api.imgur.com/3/image/{file_response['data']['deletehash']}"

    payload={}

    files=[]

    headers = {
        'Authorization': 'Client-ID 52d1141cb13f609'
    }

    response = requests.request("DELETE", url, headers=headers, data=payload, files=files)

    return response.json()

def get_url(file_response):
    url = file_response['data']['link']
    url.replace("\\", '')
    return url

cont_img = None
style_img = None



if content_file is not None or style_file is not None:
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
        if content_file is not None:
            cont_img = Image.open(content_file)
            st.image(cont_img,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">Style Reference</p>',unsafe_allow_html=True)
        if style_file is not None:
            style_img = Image.open(style_file)
            st.image(style_img,width=300)  

transformed_img = None
transformation_time = None

if st.button("Transform"):
    if(cont_img is None or style_img is None):
        st.error("Please upload both the images")
        st.stop()
    else:
        # picklefile = open("vae_model.pkl", "rb")
        model = model_options[selected_model]
        # pickle.load(picklefile)
        if model!=None:
            with st.spinner("Adding style✨ to your image ..."):
                t0 = time.time()
                transformed_img = model.transform_image(cont_img, style_img)
                t1 = time.time()
                transformation_time = t1-t0

        else:
            st.error("Model not available yet, we're trying our best to get it running..")
        transforming = False
        # picklefile.close()

if transformed_img is not None:
    st.subheader('Transformed Image')
    st.image(transformed_img,width=300, caption=f"generated in {transformation_time:.1f} seconds")
