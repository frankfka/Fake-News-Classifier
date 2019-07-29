import pandas as pd
import os


def get_file_str(file_path):
    """
    Get string content from file
    """
    with open(file_path, "r") as file:
        return file.read()


def load_json_data(json_path=None, pickle_path=None, pickle_to=None):
    """
    Load the JSON data file - all file paths are absolute
    Columns: claim, claimant, date, label, related_articles, indexed by id
        json_path: path to the json data file
        pickle_path: path to the dataframe pickle file
        pickle_to: specify with json_path if you want to save the dataframe to a pickle file
    """
    if pickle_path is not None:
        return pd.read_pickle(pickle_path)
    elif json_path is not None:
        json_data_str = get_file_str(json_path)
        claims_df = pd.read_json(json_data_str).set_index('id')
        if pickle_to is not None:
            claims_df.to_pickle(pickle_to)
        return claims_df
    raise ValueError("Improper arguments to function")


def load_article_data(articles_dir_path=None, pickle_path=None, pickle_to=None):
    """
    Load all articles in the given directory - all file paths are absolute
    Columns: text, indexed by id
        json_path: path to the json data file
        pickle_path: path to the dataframe pickle file
        pickle_to: specify with json_path if you want to save the dataframe to a pickle file
    """
    if pickle_path is not None:
        return pd.read_pickle(pickle_path)
    elif articles_dir_path is not None:
        article_filenames = os.listdir(articles_dir_path)
        article_strings = [get_file_str(os.path.join(articles_dir_path, file_name)) for file_name in article_filenames]
        article_ids = [os.path.splitext(filename)[0] for filename in article_filenames]
        articles_df = pd.DataFrame(data={'text': article_strings}, index=article_ids)
        if pickle_to is not None:
            articles_df.to_pickle(pickle_to)
        return articles_df
    raise ValueError("Improper arguments to function")
