import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/train_model/my_model1.h5"

#load pretrain model
model = tf.keras.models.load_model(model_path)

#load the class name
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path,target_size = (224,224)):
    #load the image
    img = Image.open(image_path)
    #resize the image
    img_array = img.resize(target_size)
    #add batch dimention 
    img_array = np.expand_dims(img_array,axis = 0)

    #scale the image values to [0,1]

    img_array = img_array.astype('float32')/255

    return img_array

def predict_img_clss(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predict_class_index = np.argmax(prediction,axis=1)[0]
    # Reverse mapping: index → class name
    index_to_class = {v: k for k, v in class_indices.items()}

    predicted_class_name = index_to_class[predict_class_index]
    return predicted_class_name
#stream lit app

st.title('🌿 Plant Life')

uplad_img = st.file_uploader("Upload your image.... ",type=["jpg","jpeg","png"])

if uplad_img is not None:
    image = Image.open(uplad_img)
    col1,col2, = st.columns(2)

    with col1:
        resize_img = image.resize((150,150))
        st.image(resize_img)

    with col2:
        if st.button('Search🦠'):
            prediction = predict_img_clss(model,uplad_img,class_indices)
            st.success(f'Prediction: {str(prediction)}')

    





