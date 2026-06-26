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
DATA_FILE_ID = "1qYTF8uR6kLAzU4ZFd5D5DSL22IoycnWT"

MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID }"
DATA_URL = f"https://drive.google.com/uc?id={DATA_FILE_ID}"

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
books_df.columns = [
    c.replace("-", "_").replace(" ", "_")
    for c in books_df.columns
]
# ======================================
# RECOMMENDATION FUNCTION
# ======================================

def recommend_books(user_id, top_n=10):

    predictions = []

    progress = st.progress(0)

    total_books = len(books_df)

    for i, row in enumerate(books_df.itertuples(index=False)):

        try:
            pred = model.predict(user_id, row.ISBN)

            predictions.append(
                (
                    row.ISBN,
                    row._asdict().get("Book-Title", getattr(row, "Book_Title", "")),
                    row._asdict().get("Book-Author", getattr(row, "Book_Author", "")),
                    row._asdict().get("Image-URL-L", getattr(row, "Image_URL_L", "")),
                    pred.est
                )
            )

        except Exception:
            pass

        if i % 500 == 0:
            progress.progress(min((i + 1) / total_books, 1.0))

    progress.empty()

    predictions.sort(
        key=lambda x: x[4],
        reverse=True
    )

    return predictions[:top_n]
# ======================================
# RECOMMEND BUTTON
# ======================================

if st.button("📚 Recommend Books"):

    if user_id <= 0:
        st.error("Please enter a valid User ID.")

    else:

        with st.spinner("Finding the best books for you..."):

            recommendations = recommend_books(user_id)

        if len(recommendations) == 0:
            st.warning("No recommendations found.")

        else:

            st.success("Top Recommended Books")

            for i, book in enumerate(recommendations, start=1):

                isbn = book[0]
                title = book[1]
                author = book[2]
                image = book[3]
                score = round(book[4], 2)

                col1, col2 = st.columns([1, 4])

                with col1:

                    if pd.notna(image) and str(image).startswith("http"):
                        st.image(image, width=120)
                    else:
                        st.write("📕")

                with col2:

                    st.markdown(f"### {i}. {title}")

                    st.write(f"**Author:** {author}")

                    st.write(f"**ISBN:** {isbn}")

                    st.write(f"**Predicted Rating:** ⭐ {score}")

                st.divider()

# ======================================
# FOOTER
# ======================================

st.markdown("---")
st.caption("Book Recommendation System using Surprise SVD")
st.success("Everything loaded successfully!")

# ======================================
# USER INPUT
# ======================================

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)
