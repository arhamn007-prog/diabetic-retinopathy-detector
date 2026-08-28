
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "best_model.keras"

model = load_model(MODEL_PATH)

# ==============================
# CLASS NAMES
# ==============================

class_names = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# ==============================
# PAGE DESIGN
# ==============================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>
    .stButton > button {
        background-color: #0EA5A4;
        color: white;
        border-radius: 10px;
        border: none;
    }

    [data-testid="stFileUploader"] {
        border: 2px solid #0EA5A4;
        border-radius: 12px;
        padding: 10px;
    }

    h1, h2, h3 {
        color: #14B8A6;
    }
</style>
""", unsafe_allow_html=True)
st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to get a 5-class DR prediction.")

st.divider()

# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload Retina Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Retinal Image")
        st.image(image, use_container_width=True)

    # Resize exactly like training
    processed_image = image.resize((224, 224))

    img_array = np.array(processed_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_class = int(np.argmax(prediction))
    confidence = float(prediction[predicted_class]) * 100

    with col2:

        st.subheader("Diagnosis")

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for i in range(5):
            probability = float(prediction[i]) * 100

            st.write(
                f"**{class_names[i]}:** {probability:.2f}%"
            )

            st.progress(min(probability / 100, 1.0))

else:
    st.info("Please upload a retinal fundus image to begin.")
