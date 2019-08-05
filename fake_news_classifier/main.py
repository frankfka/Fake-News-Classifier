import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt

import fake_news_classifier.util.io_util as io_util
import fake_news_classifier.model.two_to_one_lstm as two_to_one_lstm
import fake_news_classifier.model.single_input_lstm as single_input_lstm
import fake_news_classifier.model.two_to_one_seq2seq_lstm as two_to_one_seq2seq_lstm
import fake_news_classifier.model.two_to_one_nodense_lstm as two_to_one_nodense_lstm
from fake_news_classifier.model import two_to_one_seq2seq_lstm_2


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

    # Trim to 5000 entries
    json_df = json_df.iloc[10001:15000, :]
    return json_df, articles_df


def get_vectorized_trainable_df(trainable_df, spacy_nlp):
    """
    Given a trainable DF (claim, support, label), create a DF that vectorizes claim and support strings
    """
    claims = trainable_df.loc[:, 'claim']
    supporting_evidence = trainable_df.loc[:, 'support']
    labels = trainable_df.loc[:, 'label']

    vector_claims = [vectorize(claim, spacy_nlp) for claim in claims]
    vector_support = [vectorize(support, spacy_nlp) for support in supporting_evidence]
    return pd.DataFrame(data={"claim": vector_claims, "support": vector_support, "label": labels})


def vectorize(text, nlp):
    """
    Uses Spacy to convert a string into a sequence of tokens
    """
    spacy_doc = nlp(text)
    return [token.vector for token in spacy_doc]


# Normalize all the counts
def normalize_claim_counts(df):
    """
    Strips out extra claims so we have a balanced dataset - i.e. # of labels for 0,1,2 are the same
    """
    true_claims = df[df['label'] == 2]
    neutral_claims = df[df['label'] == 1]
    false_claims = df[df['label'] == 0]
    max_index = min([
        len(true_claims.index),
        len(neutral_claims.index),
        len(false_claims.index),
    ])
    return pd.concat([
        true_claims[0: max_index],
        neutral_claims[0: max_index],
        false_claims[0: max_index]
    ]).sample(frac=1)  # This shuffles


def plot_history(training_history, with_validation):
    # Accuracy Plot
    plt.plot(training_history.history['acc'])
    if with_validation:
        plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Loss Plot
    plt.plot(training_history.history['loss'])
    if with_validation:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

CURRENT_DIR = os.getcwd()

RELATIVE_DATA_DIR = '../data/'
MASTER_JSON_FILE_NAME = 'train.json'
ARTICLES_FOLDER_NAME = 'train_articles'

JSON_DATA_PICKLE = 'checkpoints/json_data.pkl'
ARTICLES_DATA_PICKLE = 'checkpoints/articles_data.pkl'

TRAINABLE_DATA_PICKLE = 'checkpoints/trainable_data_10000_15000.pkl'
TRAINABLE_VECTORIZED_DATA_PICKLE = 'checkpoints/trainable_vectorized_data.pkl'

# Uncomment below if vectorizing/creating trainable DF
# spacy_nlp = spacy.load('en_core_web_md')

'''
Load JSON and Articles data as separate DF's and create a trainable DF
'''
# json_df, articles_df = load_data()
# trainable_df = get_trainable_df(json_df, articles_df, spacy_nlp)
# trainable_df.to_pickle(os.path.join(CURRENT_DIR, TRAINABLE_DATA_PICKLE))
# print(trainable_df.head())

'''
Load Trainable DF and vectorize it
'''
# trainable_df = pd.read_pickle(os.path.join(CURRENT_DIR, TRAINABLE_DATA_PICKLE))
# vectorized_trainable_df = get_vectorized_trainable_df(trainable_df, spacy_nlp)
# vectorized_trainable_df.to_pickle(os.path.join(CURRENT_DIR, TRAINABLE_VECTORIZED_DATA_PICKLE))

'''
Load Vectorized DF
'''
vectorized_trainable_df = pd.read_pickle(os.path.join(CURRENT_DIR, TRAINABLE_VECTORIZED_DATA_PICKLE))
print(f"Number of initial data points: {len(vectorized_trainable_df.index)}")
print(vectorized_trainable_df['label'].value_counts())
normalized_vec_trainable_df = normalize_claim_counts(vectorized_trainable_df)

vector_claims = normalized_vec_trainable_df.loc[:, 'claim']
vector_support = normalized_vec_trainable_df.loc[:, 'support']
labels = normalized_vec_trainable_df.loc[:, 'label']


'''
Train Model
'''
history = single_input_lstm.train(vector_claims, vector_support, labels)
plot_history(history, True)