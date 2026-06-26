from flask import Flask, render_template, request
import os
import joblib
import gdown

app = Flask(__name__)

# ==========================
# MODEL CONFIGURATION
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")

# Google Drive File ID
FILE_ID = "1Ym4XMLzHIy-6Nwz7O3fykZQDfWMG-9tu"

# Download URL
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ==========================
# DOWNLOAD MODEL IF NEEDED
# ==========================
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    print("Model downloaded successfully!")

# ==========================
# LOAD MODEL
# ==========================
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# RECOMMENDATION PAGE
# ==========================
@app.route("/recommend", methods=["POST"])
def recommend():

    # Example input
    user_id = int(request.form["user_id"])

    # ----------------------------------------------------
    # Replace this section with YOUR recommendation logic
    # Example:
    #
    # recommendations = get_recommendations(user_id)
    #
    # ----------------------------------------------------

    recommendations = [
        "Book 1",
        "Book 2",
        "Book 3",
        "Book 4",
        "Book 5"
    ]

    return render_template(
        "index.html",
        recommendations=recommendations
    )


# ==========================
# RUN APPLICATION
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
