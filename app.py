import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('Skin_Cancer.h5')

def predict_skin_cancer(uploaded_file, model):
    img = Image.open(uploaded_file).resize((224, 224))  # fix: open directly from file object
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    return class_label, img

st.title("Skin Cancer Detection")
st.markdown("""
    This is a skin cancer detection application. Upload an image, and the model will predict 
    whether the skin lesion is **Malignant** or **Benign**.
""")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    class_label, img = predict_skin_cancer(uploaded_image, model)
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    if class_label == "Malignant":
        st.error(f"⚠️ Prediction: {class_label}")
    else:
        st.success(f"✅ Prediction: {class_label}")

st.markdown("""
    ### About the Model
    This model uses CNN architecture to classify skin lesions as **Benign** or **Malignant**.
    #### How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.
""")
