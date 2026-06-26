import os
import joblib
import gdown
import pandas as pd
import streamlit as st

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System")
st.write("Get personalized book recommendations using Machine Learning.")

# ======================================
# FOLDERS
# ======================================
MODEL_DIR = "models"
DATA_DIR = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "svd_modelnn.pkl")
DATA_PATH = os.path.join(DATA_DIR, "books_metadata.csv")

# ======================================
# GOOGLE DRIVE FILE IDS
# ======================================

MODEL_FILE_ID = "1Ym4XMLzHIy-6Nwz7O3fykZQDfWMG-9tu"

# Replace with your metadata Google Drive File ID
DATA_FILE_ID = "YOUR_METADATA_FILE_ID"

MODEL_URL = f"https://drive.google.com/uc?id={1Ym4XMLzHIy-6Nwz7O3fykZQDfWMG-9tu}"
DATA_URL = f"https://drive.google.com/uc?id={1qYTF8uR6kLAzU4ZFd5D5DSL22IoycnWT}"

# ======================================
# DOWNLOAD FILE FUNCTION
# ======================================

def download_file(url, destination, name):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {name}..."):
            gdown.download(url, destination, quiet=False)
        st.success(f"{name} downloaded successfully!")

# Download model
download_file(MODEL_URL, MODEL_PATH, "SVD Model")

# Download metadata
download_file(DATA_URL, DATA_PATH, "Book Metadata")

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================================
# LOAD BOOK METADATA
# ======================================

@st.cache_data
def load_books():
    return pd.read_csv(DATA_PATH)

books_df = load_books()

st.success("Everything loaded successfully!")

# ======================================
# USER INPUT
# ======================================

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)
