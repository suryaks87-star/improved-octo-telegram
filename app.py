from flask import Flask, render_template, request
import os
import joblib
import gdown

app = Flask(__name__)

# ============================
# Model Configuration
# ============================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "svd_modelnn.pkl")

# Replace this with your Google Drive File ID
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

DOWNLOAD_URL = f"https://drive.google.com/uc?id={1Ym4XMLzHIy-6Nwz7O3fykZQDfWMG-9tu}"

# ============================
# Download Model if Not Exists
# ============================
if not os.path.exists(MODEL_PATH):
    print("Model not found.")
    print("Downloading model from Google Drive...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    print("Model downloaded successfully!")

# ============================
# Load Model
# ============================
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# ============================
# Home Page
# ============================
@app.route("/")
def home():
    return render_template("index.html")


# ============================
# Recommendation Route
# ============================
@app.route("/recommend", methods=["POST"])
def recommend():

    user_id = int(request.form["user_id"])

    # ---------------------------------------
    # Add your recommendation logic here
    # Example:
    #
    # predictions = ...
    #
    # return render_template(
    #     "index.html",
    #     recommendations=predictions
    # )
    # ---------------------------------------

    return render_template(
        "index.html",
        recommendations=["Recommendation logic goes here."]
    )


# ============================
# Run App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
