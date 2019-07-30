import os
import pandas as pd
import spacy
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

import fake_news_classifier.util.io_util as io_util

CURRENT_DIR = os.getcwd()

RELATIVE_DATA_DIR = '../data/'
MASTER_JSON_FILE_NAME = 'train.json'
ARTICLES_FOLDER_NAME = 'train_articles'

JSON_DATA_PICKLE = 'checkpoints/json_data.pkl'
ARTICLES_DATA_PICKLE = 'checkpoints/articles_data.pkl'

TRAINABLE_DATA_PICKLE = 'checkpoints/trainable_data.pkl'
TRAINABLE_VECTORIZED_DATA_PICKLE = 'checkpoints/trainable_vectorized_data.pkl'


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

    # Trim to first 5000 entries
    json_df = json_df.iloc[0:5000, :]
    return json_df, articles_df


def vectorize(text, nlp):
    """
    Uses Spacy to convert a string into a sequence of tokens
    """
    spacy_doc = nlp(text)
    return [token.vector for token in spacy_doc]

# json_df, articles_df = load_data()
# trainable_df = get_trainable_df(json_df, articles_df)
# trainable_df.to_pickle(os.path.join(CURRENT_DIR, TRAINABLE_DATA_PICKLE))
# print(trainable_df.head())


MAX_SEQ_LEN = 500  # Maximum word length of each support/claim - sequences will be padded to this length
WORD_VECTOR_SIZE = 300  # Size of each word vector

trainable_df = pd.read_pickle(os.path.join(CURRENT_DIR, TRAINABLE_DATA_PICKLE))
claims = trainable_df.loc[:, 'claim']
supporting_evidence = trainable_df.loc[:, 'support']
labels = trainable_df.loc[:, 'label']

spacy_nlp = spacy.load('en_core_web_md')

print('Vectorizing')
vector_claims = [vectorize(claim, spacy_nlp) for claim in claims]
vector_support = [vectorize(support, spacy_nlp) for support in supporting_evidence]
vectorized_trainable_df = pd.DataFrame(data={"claim": vector_claims, "support": vector_support, "label": labels})
vectorized_trainable_df.to_pickle(os.path.join(CURRENT_DIR, TRAINABLE_VECTORIZED_DATA_PICKLE))

print('Padding Sequences')
vector_claims = pad_sequences(vector_claims, maxlen=MAX_SEQ_LEN, truncating='post', padding='post', dtype='float32')
vector_support = pad_sequences(vector_support, maxlen=MAX_SEQ_LEN, truncating='post', padding='post', dtype='float32')
categorical_labels = to_categorical(labels, num_classes=3)

'''
Define LSTM for Claims
'''
claims_input_model = Sequential()
claims_input_model.add(LSTM(units=64, input_shape=(MAX_SEQ_LEN, WORD_VECTOR_SIZE), dropout=0.3, recurrent_dropout=0.3))
claims_input_model.add(Dense(32, activation='relu'))
# Batch normalization?

'''
Define LSTM for Support
'''
support_input_model = Sequential()
support_input_model.add(LSTM(units=64, input_shape=(MAX_SEQ_LEN, WORD_VECTOR_SIZE), dropout=0.3, recurrent_dropout=0.3))
support_input_model.add(Dense(32, activation='relu'))

'''
Define Model After Concatenating support + claim
'''
mergedModel = Concatenate()([claims_input_model.output, support_input_model.output])
mergedModel = Dense(8, activation='relu')(mergedModel)
mergedModel = Dense(3, activation='softmax')(mergedModel)

'''
Define the entire model from sub-models
'''
complete_model = Model([claims_input_model.input, support_input_model.input], mergedModel)
complete_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
complete_model.summary()
history = complete_model.fit(
    [vector_claims, vector_support],
    categorical_labels,
    batch_size=1, epochs=100, verbose=1, validation_split=0.2
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

