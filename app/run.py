import json
import re

import nltk
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar

import joblib
from sqlalchemy import create_engine

nltk.download(["punkt", "stopwords"])

app = Flask(__name__)


def tokenize(text):
    """Normalize, tokenize and stem text string

    Args:
        text (str): String containing message for processing

    Returns:
        stemmed (list[str]). List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)

    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return stemmed


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df_raw = pd.read_sql_table("messages", engine)
df_model = df_raw.drop("child_alone", axis=1)
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    ## Data for figure 1: Distribution of Message Genres
    genre_counts = df_raw.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    ## Stacked Bar Chart
    df_fig2 = (
        df_raw.iloc[:, 4:].apply(lambda x: x.value_counts(normalize=False)).fillna(0).T
    )

    trace1 = {
        "x": df_fig2.index.tolist(),
        "y": df_fig2[0],
        "name": "val_0",
        "type": "bar",
    }

    trace2 = {
        "x": df_fig2.index.tolist(),
        "y": df_fig2[1],
        "name": "val_1",
        "type": "bar",
    }

    # create visuals
    graphs = [
        # Fig1
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
            },
        },
        # Fig 2
        {
            "data": [trace1, trace2],
            "layout": {
                "barmode": "stack",
                "title": "Value Counts of Category Variables",
                "yaxis": {"title": "Count"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_model.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
