import os
import joblib
import gdown
import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Book Recommendation System")

st.title("📚 Book Recommendation System")

# ==========================
# MODEL CONFIGURATION
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svd_modelnn.pkl")

FILE_ID = "1Ym4XMLzHIy-6Nwz7O3fykZQDfWMG-9tu"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ==========================
# DOWNLOAD MODEL
# ==========================
if not os.path.exists(MODEL_PATH):

    st.info("Downloading model...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    st.success("Model downloaded!")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.success("Model Loaded Successfully!")

# ==========================
# USER INPUT
# ==========================
user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

# ==========================
# RECOMMEND BUTTON
# ==========================
if st.button("Recommend Books"):

    # Replace with your recommendation function
    recommendations = [
        "Book 1",
        "Book 2",
        "Book 3",
        "Book 4",
        "Book 5"
    ]

    st.subheader("Recommended Books")

    for book in recommendations:
        st.write("✅", book)
