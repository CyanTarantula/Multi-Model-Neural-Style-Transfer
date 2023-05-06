import streamlit as st
import pandas as pd
import numpy as np
from  PIL import Image, ImageEnhance
import pickle
from streamlit import config
import style_transfer_functions

st.sidebar.title("Configurations")
st.title("Style Transfer using Different Architectures")

## Sidebar
with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to convert your normal Images into different styles.\nUpload an Image that you want to transform, and another Image which is the Style reference. What you get as a result is the original image stylized according to the reference style Image.\n\nThis app was created by Soumik, Yash and Stuti as a part of the DLOps project for the course CSL4020: Deep Learning offered at IIT Jodhpur during Jan-May 2023.
     """)

st.sidebar.header("Select Model Architecture")
model_options = {
    'Variational AE': 1,
    'Tranformer' : 2,
    'Styl-GAN' : 3,
    'Pics-Art API' : 4,
}
selected_model = st.sidebar.radio("Models",tuple(model_options.keys()))


## Main Content
st.subheader('Upload your Image and Style Reference')
content_file = st.file_uploader("Image to Style", type=['jpg','png','jpeg'])
stlye_file = st.file_uploader("Style Reference Image", type=['jpg','png','jpeg'])

#Add 'Original' and 'Style' columns

cont_img = None
style_img = None

if content_file is not None or stlye_file is not None:
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Original</p>',unsafe_allow_html=True)
        if content_file is not None:
            cont_img = Image.open(content_file)
            st.image(cont_img,width=300)  
            # st.write(
            #     image_to_url(
            #         image=cont_img,
            #         width=300,
            #         clamp=False,
            #         channels="RGB",
            #         output_format="auto",
            #         image_id=,  # each uploaded file has a file.id
            #     )
            # )

    with col2:
        st.markdown('<p style="text-align: center;">Style Reference</p>',unsafe_allow_html=True)
        if stlye_file is not None:
            style_img = Image.open(stlye_file)
            st.image(style_img,width=300)  

transformed_img = None

if st.button("Transform"):
    if(cont_img is None or style_img is None):
        st.error("Please upload both the images")
        st.stop()
    else:
        st.warning('Transforming Image')
        # picklefile = open("vae_model.pkl", "rb")
        model = style_transfer_functions.VAE()
        # pickle.load(picklefile)
        transformed_img = model.transform_image(cont_img, style_img)
        # picklefile.close()

if transformed_img is not None:
    st.subheader('Transformed Image')
    st.image(transformed_img,width=300)
