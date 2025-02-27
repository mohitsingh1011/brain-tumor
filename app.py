# import streamlit as st 
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# from PIL import Image
# import numpy as np
# from tensorflow.keras import backend as K
# from keras.saving import register_keras_serializable

# # Constants
# CLASS_IM_HEIGHT = 150
# CLASS_IM_WIDTH = 150
# SEG_IM_HEIGHT = 256
# SEG_IM_WIDTH = 256
# SMOOTH = 100
# LABELS = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# epsilon = 1e-5
# smooth = 1

# def tversky(y_true, y_pred):
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#     alpha = 0.7
#     return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

# def focal_tversky(y_true,y_pred):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
    
#     pt_1 = tversky(y_true, y_pred)
#     gamma = 0.75
#     return K.pow((1-pt_1), gamma)

# def tversky_loss(y_true, y_pred):
#     return 1 - tversky(y_true,y_pred)


# # Load models
# @st.cache_resource
# def load_classification_model():
#     return tf.keras.models.load_model('./effnet.h5')

# @st.cache_resource
# def load_segmentation_model():
#     return load_model('seg_model.h5', custom_objects={
#             "focal_tversky": focal_tversky, 
#             "tversky": tversky, 
#             "tversky_loss": tversky_loss
#         })

# model_class = load_classification_model()
# model_seg = load_segmentation_model()

# # App header
# st.title("\U0001F5E3 Brain Tumor Detection and Classification")
# st.write("Upload an MRI image to classify and segment for brain tumors.")

# # File uploader
# file = st.file_uploader("Upload MRI Image", type=["jpg", "png"], label_visibility="visible")

# def upload_predict(image, model):
#     size = (CLASS_IM_WIDTH, CLASS_IM_HEIGHT)
#     img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
#     img_reshaped = img_resized[np.newaxis, ...]
#     pred = model.predict(img_reshaped)
#     return np.argmax(pred)

# if file:
#     # Display uploaded image
#     image = Image.open(file)
#     st.image(image, caption="Uploaded MRI Image", width=246)

#     # Preprocess for segmentation
#     img_array = np.asarray(image)
#     img_resized = cv2.resize(img_array, (SEG_IM_WIDTH, SEG_IM_HEIGHT))
#     img_normalized = img_resized / 255.0
#     img_input = img_normalized[np.newaxis, ...]

#     with st.spinner("Processing image..."):
#         # Segmentation
#         seg_pred = model_seg.predict(img_input)
#         seg_mask = (np.squeeze(seg_pred) > 0.5).astype('uint8')  # Binary mask (0 or 1)

#         # Resize mask back to original image size
#         seg_mask_resized = cv2.resize(seg_mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)

#         # Apply prominent colored overlay to the original image
#         color_overlay = np.zeros_like(img_array, dtype='uint8')
#         color_overlay[seg_mask_resized == 1] = [255, 0, 0]  # Bright red for tumor region
#         alpha = 0.6  # Increased transparency for better visibility
#         masked_img = cv2.addWeighted(img_array, 1 - alpha, color_overlay, alpha, 0)

#         # Classification
#         class_pred = LABELS[upload_predict(image, model_class)]

#     # Display results side by side
#     st.write("## Results")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(img_array, caption="Original Image", use_container_width=True)
#     if class_pred != "No Tumor":
#         with col2:
#             st.image(masked_img, caption="Segmented Image", use_container_width=True)

#     st.success(f"The image is classified as: **{class_pred}**")
# else:
#     st.info("Please upload an image file to proceed.")




import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

# Constants
CLASS_IM_HEIGHT = 150
CLASS_IM_WIDTH = 150
SEG_IM_HEIGHT = 256
SEG_IM_WIDTH = 256
SMOOTH = 100
LABELS = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

# Load models
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model('./effnet.h5')

@st.cache_resource
def load_segmentation_model():
    return load_model('seg_model.h5', custom_objects={
            "focal_tversky": focal_tversky, 
            "tversky": tversky, 
            "tversky_loss": tversky_loss
        })

model_class = load_classification_model()
model_seg = load_segmentation_model()

# App header
st.title("\U0001F5E3 Brain Tumor Detection and Classification")
st.write("Upload an MRI image to classify and segment for brain tumors.")

# File uploader
file = st.file_uploader("Upload MRI Image", type=["jpg", "png"], label_visibility="visible")

def upload_predict(image, model):
    size = (CLASS_IM_WIDTH, CLASS_IM_HEIGHT)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img_reshaped = img_resized[np.newaxis, ...]
    pred = model.predict(img_reshaped)
    return np.argmax(pred)

if file:
    # Display uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded MRI Image", width=246)

    # Preprocess for segmentation
    img_array = np.asarray(image)
    img_resized = cv2.resize(img_array, (SEG_IM_WIDTH, SEG_IM_HEIGHT))
    img_normalized = img_resized / 255.0
    img_input = img_normalized[np.newaxis, ...]

    with st.spinner("Processing image..."):
        # Classification
        class_pred_index = upload_predict(image, model_class)
        class_pred = LABELS[class_pred_index]
    
        if class_pred == "No Tumor":
            st.success("The image was classified as 'No Tumor'. Please upload a valid brain MRI image if this seems incorrect.")
        elif class_pred not in LABELS:
            st.error("Invalid image. Please upload a valid brain MRI image.")
        else:
            # Segmentation
            seg_pred = model_seg.predict(img_input)
            seg_mask = (np.squeeze(seg_pred) > 0.5).astype('uint8')  # Binary mask (0 or 1)

            # Resize mask back to original image size
            seg_mask_resized = cv2.resize(seg_mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Apply prominent colored overlay to the original image
            color_overlay = np.zeros_like(img_array, dtype='uint8')
            color_overlay[seg_mask_resized == 1] = [255, 0, 0]  # Bright red for tumor region
            alpha = 0.6  # Increased transparency for better visibility
            masked_img = cv2.addWeighted(img_array, 1 - alpha, color_overlay, alpha, 0)

            # Display results
            st.write("## Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_container_width=True)
            if class_pred != "No Tumor":
                with col2:
                    st.image(masked_img, caption="Segmented Image", use_container_width=True)

            st.success(f"The image is classified as: **{class_pred}**")
else:
    st.info("Please upload an image file to proceed.")
