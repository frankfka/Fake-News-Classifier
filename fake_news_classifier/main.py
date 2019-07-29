import os
import pandas as pd
from fake_news_classifier.util.preprocess_util import get_trainable_df
import fake_news_classifier.util.io_util as io_util

CURRENT_DIR = os.getcwd()

RELATIVE_DATA_DIR = '../data/'
MASTER_JSON_FILE_NAME = 'train.json'
ARTICLES_FOLDER_NAME = 'train_articles'

JSON_DATA_PICKLE = 'checkpoints/json_data.pkl'
ARTICLES_DATA_PICKLE = 'checkpoints/articles_data.pkl'

TRAINABLE_DATA_PICKLE = 'checkpoints/trainable_data.pkl'


def load_data():
    """
    Load the Data
    - Claims & Metadata in 1 table
    - Articles in another table with their parsed text
    """
    data_dir = os.path.join(CURRENT_DIR, RELATIVE_DATA_DIR)
    json_file_path = os.path.join(data_dir, MASTER_JSON_FILE_NAME)
    articles_file_path = os.path.join(data_dir, ARTICLES_FOLDER_NAME)
    json_pickle_path = os.path.join(CURRENT_DIR, JSON_DATA_PICKLE)
    articles_pickle_path = os.path.join(CURRENT_DIR, ARTICLES_DATA_PICKLE)

    # Loads from raw files
    # json_df = io_util.load_json_data(json_path=json_file_path, pickle_to=json_pickle_path)
    # articles_df = io_util.load_article_data(articles_dir_path=articles_file_path, pickle_to=articles_pickle_path)

    json_df = io_util.load_json_data(pickle_path=json_pickle_path)
    articles_df = io_util.load_article_data(pickle_path=articles_pickle_path)
    return json_df, articles_df


trainable_df = pd.read_pickle(os.path.join(CURRENT_DIR, TRAINABLE_DATA_PICKLE))
for idx, row in trainable_df.iterrows():
    print(f"INDEX: {idx}")
    print(row['claim'])
    print(row['support'])