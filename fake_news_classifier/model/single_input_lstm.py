from keras import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from fake_news_classifier.eval.f1_callback import F1KerasCallback

MAX_SEQ_LEN = 500  # Maximum word length of each support/claim - sequences will be padded to this length
WORD_VECTOR_SIZE = 300  # Size of each word vector
NUM_EPOCHS = 10
BATCH_SIZE = 50
VALIDATION_SPLIT = 0.2


def train(vector_claims, vector_support, labels):
    """
    Trains with given data and returns history
    """
    vector_claims = pad_sequences(vector_claims, maxlen=MAX_SEQ_LEN, dtype='float32')
    vector_support = pad_sequences(vector_support, maxlen=MAX_SEQ_LEN, dtype='float32')
    # double WORD_VECTOR_SIZE (i.e. add vectors column-wise)
    vector_in = np.array([np.hstack((claim, support)) for claim, support in zip(vector_claims, vector_support)])
    # double SEQ_LEN (i.e. add vectors row-wise)
    # vector_in = np.hstack((vector_claims, vector_support))
    categorical_labels = to_categorical(labels, num_classes=3)
    complete_model = model(MAX_SEQ_LEN, WORD_VECTOR_SIZE * 2)
    complete_model.summary()
    return complete_model.fit(
        vector_in,
        categorical_labels,
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT,
        callbacks=[F1KerasCallback()]
    )


def model(max_seq_len, word_vec_size):
    """
    Single LSTM input network
    """
    nn = Sequential()
    nn.add(LSTM(units=64, input_shape=(max_seq_len, word_vec_size), dropout=0.5, recurrent_dropout=0.5))
    nn.add(BatchNormalization())
    nn.add(Dense(32, activation='relu'))
    nn.add(Dropout(0.3))
    nn.add(Dense(3, activation='softmax'))
    nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return nn
