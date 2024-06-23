import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def read_image(file):
    x = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    ori_x = x.copy()
    x = cv2.resize(x, (W, H)) / 255.0
    return ori_x, x

def read_mask(file):
    x = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    ori_x = x.copy()
    x = cv2.resize(x, (W, H)) / 255.0
    return ori_x, x

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def predict_mask(model, image):
    x = cv2.resize(image, (W, H)) / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)[0]
    y_pred = (y_pred > 0.5).astype(np.int32)
    return np.squeeze(y_pred, axis=-1)

def main():
    st.title("Image Segmentation Prediction")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg"])
    uploaded_mask = st.file_uploader("Upload a mask", type=["jpg"])  # Assuming masks are grayscale images

    if uploaded_image is not None and uploaded_mask is not None:

        ori_x, x = read_image(uploaded_image)
        ori_y, y = read_mask(uploaded_mask)
        st.image(ori_x, caption='Uploaded Image', use_column_width=True)
        st.image(ori_y, caption='Uploaded Mask', use_column_width=True)

        # Load the model
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model("files/model.h5")

        # Make prediction
        predicted_mask = predict_mask(model, x * 255)  # Scale x back to 0-255 range
        save_image(predicted_mask * 255, "predicted.jpg")
        st.image(predicted_mask * 255, caption='Predicted Mask', use_column_width=True)

if __name__ == "__main__":
    main()
