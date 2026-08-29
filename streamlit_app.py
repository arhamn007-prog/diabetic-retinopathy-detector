
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from supabase import create_client
# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background-color: #0F172A;
    color: #E2E8F0;
}

h1, h2, h3 {
    color: #5EEAD4 !important;
    font-weight: 700;
}

p, label {
    color: #CBD5E1 !important;
}

.stButton > button {
    background-color: #0F766E;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #115E59;
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border: 2px solid #99F6E4;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #CCFBF1;
    border-radius: 12px;
    padding: 15px;
}

hr {
    border-color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)
# ==============================
# SUPABASE CONNECTION
# ==============================

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)# ==============================
# LOGIN / SIGNUP
# ==============================

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:

    st.title("🩺 Diabetic Retinopathy Detection")
    st.subheader("Login / Sign Up")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": login_email,
                    "password": login_password
                })

                st.session_state.user = response.user
                st.success("Login successful!")
                st.rerun()

            except Exception as e:
                st.error("Invalid email or password.")

    with tab2:
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        if st.button("Create Account"):
            try:
                response = supabase.auth.sign_up({
                    "email": signup_email,
                    "password": signup_password
                })

                st.success(
                    "Account created! Please check your email to verify your account."
                )

            except Exception as e:
                st.error(str(e))

    st.stop()

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
