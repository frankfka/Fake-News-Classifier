from keras import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 500  # Maximum word length of each support/claim - sequences will be padded to this length
WORD_VECTOR_SIZE = 300  # Size of each word vector
NUM_EPOCHS = 30
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.2


def train(vector_claims, vector_support, labels):
    """
    Trains with given data and returns history
    """
    vector_claims = pad_sequences(vector_claims, maxlen=MAX_SEQ_LEN, truncating='post', padding='post', dtype='float32')
    vector_support = pad_sequences(vector_support, maxlen=MAX_SEQ_LEN, truncating='post', padding='post',
                                   dtype='float32')
    categorical_labels = to_categorical(labels, num_classes=3)
    complete_model = model(MAX_SEQ_LEN, WORD_VECTOR_SIZE)
    complete_model.summary()
    return complete_model.fit(
        [vector_claims, vector_support],
        categorical_labels,
        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT
    )


def model(max_seq_len, word_vec_size):
    """
    Two separate LSTM's concatenated then passed through dense layers for final prediction
    """

    # TODO: Bidirectional, batch normalization, play with params
    # https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
    '''
    Define LSTM for Claims
    '''
    claims_input_model = Sequential()
    claims_input_model.add(
        LSTM(units=64, input_shape=(max_seq_len, word_vec_size), dropout=0.3, recurrent_dropout=0.3))
    claims_input_model.add(Dense(32, activation='relu'))
    # Batch normalization?

    '''
    Define LSTM for Support
    '''
    support_input_model = Sequential()
    support_input_model.add(
        LSTM(units=64, input_shape=(max_seq_len, word_vec_size), dropout=0.3, recurrent_dropout=0.3))
    support_input_model.add(Dense(32, activation='relu'))

    '''
    Define Model After Concatenating support + claim
    '''
    merged_model = Concatenate()([claims_input_model.output, support_input_model.output])
    merged_model = Dense(8, activation='relu')(merged_model)
    merged_model = Dense(3, activation='softmax')(merged_model)

    '''
    Define the entire model from sub-models
    '''
    complete_model = Model([claims_input_model.input, support_input_model.input], merged_model)
    complete_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return complete_model
