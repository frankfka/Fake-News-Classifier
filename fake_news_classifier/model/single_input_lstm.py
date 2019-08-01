from keras import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np


MAX_SEQ_LEN = 500  # Maximum word length of each support/claim - sequences will be padded to this length
WORD_VECTOR_SIZE = 300  # Size of each word vector
NUM_EPOCHS = 30
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.2


def train(vector_claims, vector_support, labels):
    """
    Trains with given data and returns history
    """
    vector_claims = pad_sequences(vector_claims, maxlen=MAX_SEQ_LEN, dtype='float32')
    vector_support = pad_sequences(vector_support, maxlen=MAX_SEQ_LEN, dtype='float32')
    # hstack will double WORD_VECTOR_SIZE (i.e. add vectors column-wise)
    # vstack will double SEQ_LEN (i.e. add vectors row-wise)
    vector_in = np.hstack((vector_claims, vector_support))  # TODO: try vstack
    categorical_labels = to_categorical(labels, num_classes=3)
    complete_model = model(MAX_SEQ_LEN, WORD_VECTOR_SIZE * 2)
    complete_model.summary()
    return complete_model.fit(
        vector_in,
        categorical_labels,
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT
    )


def model(max_seq_len, word_vec_size):
    """
    Single LSTM input network
    """
    nn = Sequential()
    nn.add(LSTM(units=256, input_shape=(max_seq_len, word_vec_size), dropout=0.3, recurrent_dropout=0.3))
    nn.add(Dense(128, activation='relu'))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(3, activation='softmax'))
    nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return nn
