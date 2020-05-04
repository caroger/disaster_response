import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data in .csv files. Store and return the data in pandas DataFrame

    Keyword arguments:
    messages_filepath -- path to the `disaster_categories.csv` file
    categories_filepath -- path to the `disaster_messages.csv` file

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # ensure both data sets have equal number of rows
    assert messages.shape[0] == categories.shape[0]
    # merge data sets
    df = pd.merge(messages, categories, on="id", how="left")
    return df


def clean_data(df):
    """Returns a cleaned and preprocessed data that is ready for
    machine learning pipeline

    keyword arguments:
    df -- DataFrame containing joined message and categories data
    """

    # create a data frame of 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # drop the original categories columns
    df = df.drop(labels="categories", axis=1)
    # use 1st row to extract a list of new column names for categories
    row = categories.loc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of 'categories'
    categories.columns = category_colnames
    # conver category values to binary
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    # remove rows where the `related` column has a value of 2
    index_drop = categories[categories["related"] == 2].index
    df.drop(index_drop, inplace=True)
    categories.drop(index_drop, inplace=True)
    # concatenante the original df with the new `categories` df
    df = pd.concat([df, categories], axis=1)
    # remove duplicated rows
    df.drop_duplicates(keep="first", inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
