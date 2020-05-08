import pickle
import re
import sys
import warnings

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(["punkt", "stopwords"])
warnings.simplefilter("ignore")


def load_data(database_filepath):
    """ Load data from database for machine learning modeling

    Args:
        database_filepath (str): path to the .db file

    Returns:
        tuple of three elements (X, Y, category_names):
           - X(DataFrame): input variables
           - Y(DataFrame): output variables
           - category_names (list[str]): output varaible names
    """
    # Load data from database
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)

    # create X, Y dataframes for machine learning
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre", "child_alone"], axis=1)
    category_names = Y.columns.tolist()

    return (X, Y, category_names)


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


def build_model(x_train, y_train):
    """Performs gridsearchCV and returns the model using the best estimator
    fitted on the full training data

    Args:
        x_train: input variable
        y_train: output variables

    Returns: the best estimator from RandomizedSearchCV
    """

    # Pipeline for SGD model training
    pipe = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(SGDClassifier())),
        ]
    )

    # Params for grid search cv
    pram_grid = {
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__norm": ("l1", "l2"),
        "clf__estimator__max_iter": (100, 500, 1000),
        "clf__estimator__alpha": (0.00001, 0.000001),
        "clf__estimator__penalty": ("l2", "elasticnet"),
        # hinge = linear SVM, log = logistic regression
        "clf__estimator__loss": ("hinge", "log"),
    }

    # Perform grid search cross validation to find the best estimator and params
    # based on f1_micro score
    cv = RandomizedSearchCV(pipe, pram_grid, scoring="f1_micro", verbose=5)
    cv.fit(x_train, y_train)

    return cv.best_estimator_


def evaluate_model(model, X_test, Y_test, category_names):
    """Print model evaluation scores (f1-micro) for each output category

    Args:
        model: sklearn estimator
        X_test (DataFrame): input variables for test
        Y_test (DataFrame): output variables for test
        category_names (list[str]): category names for the output variables

    Returns: None
    """

    Y_pred = model.predict(X_test)
    metrics_dict = classification_report(
        Y_test, Y_pred, output_dict=True, target_names=category_names
    )

    print(pd.DataFrame(metrics_dict).T[:-3])


def save_model(model, model_filepath):
    """Save model to specified filepath in .pkl format

    Args:
        model: model to be saved
        model_filepath: file path to the output pickle file

    Returns: None
    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:

        # Set Global random seed
        np.random.seed(95130)

        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model(X_train, Y_train)

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
