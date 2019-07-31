from keras import Sequential, Model
from keras.layers import LSTM, Dense, Concatenate


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
