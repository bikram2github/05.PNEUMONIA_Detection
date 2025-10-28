import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from PIL import Image
import time


@st.cache_resource
def load_trained_model():
    return load_model("chest_xray_model.keras")

model = load_trained_model()


st.warning("⚠️ This tool is for educational purposes only and should not be used for medical diagnosis.")

st.title(" Pneumonia Detection")

st.write("Upload a chest X-ray image to detect possible pneumonia.")


uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])


class_names = ["Normal", "Pneumonia"]

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="🩺 Uploaded X-ray", use_container_width=True)


    image = image.convert("L").resize((128, 128))
    image_array = np.expand_dims(np.array(image), axis=(0, -1))  # (1,128,128,1)


    with st.spinner("       🔍  Analyzing X-ray image please wait..."):
        time.sleep(1.5)
        prediction = model.predict(image_array)


    confidence = float(prediction[0][0])
    predicted_class = class_names[int(confidence > 0.5)]


    confidence_normal = 1 - confidence
    confidence_pneumonia = confidence


    st.progress(confidence_pneumonia if predicted_class == "Pneumonia" else confidence_normal)


    st.markdown("#### 📊 Confidence Breakdown:")
    col1, col2 = st.columns(2)
    col1.metric("Normal", f"{confidence_normal*100:.2f}%")
    col2.metric("Pneumonia", f"{confidence_pneumonia*100:.2f}%")



    if predicted_class == "Normal":
        st.success(f"✅ **Prediction: NORMAL**")

    else:
        st.error(f"🚨 **Prediction: PNEUMONIA**")



else:
    st.info("👆 Upload an image above to start the analysis.")

st.markdown("---")
st.caption("Developed for educational purposes only. Not for medical use.")
