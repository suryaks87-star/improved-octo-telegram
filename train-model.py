import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# Load ratings dataset
ratings = pd.read_csv("Ratings.csv", encoding="latin-1")

# Keep only required columns
ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]

# Create Surprise dataset
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(
    ratings[['User-ID', 'ISBN', 'Book-Rating']],
    reader
)

# Build full trainset
trainset = data.build_full_trainset()

# Train SVD model
print("Training SVD model...")
svd_model = SVD(random_state=42)
svd_model.fit(trainset)

# Save model
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)

print("✅ Model saved as svd_model.pkl")
