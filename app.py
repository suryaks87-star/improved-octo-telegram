from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load SVD model
with open("svd_model.pkl", "rb") as file:
    svd_model = pickle.load(file)

# Load dataset
books = pd.read_csv("Cleaned_Book_Data.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    user_id = int(request.form["user_id"])

    predictions = []

    # Predict rating for every book
    for isbn in books["ISBN"].unique():

        pred = svd_model.predict(user_id, isbn)

        predictions.append({
            "ISBN": isbn,
            "Predicted_Rating": pred.est
        })

    prediction_df = pd.DataFrame(predictions)

    # Merge with book details
    recommendation = prediction_df.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN"
    )

    recommendation = recommendation.sort_values(
        by="Predicted_Rating",
        ascending=False
    )

    recommendation = recommendation.drop_duplicates(
        subset="Book-Title"
    )

    top_books = recommendation.head(10)

    return render_template(
        "index.html",
        tables=top_books.to_dict(orient="records")
    )


if __name__ == "__main__":
    app.run(debug=True)