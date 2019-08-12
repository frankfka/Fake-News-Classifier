from keras import Sequential, Model
from keras.callbacks import TensorBoard
from keras.layers import Bidirectional, LSTM, BatchNormalization, Concatenate, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from fake_news_classifier.const import LABEL_IDX, TEXT_TWO_IDX, TEXT_ONE_IDX
from fake_news_classifier.model.FNCModel import FNCModel
from fake_news_classifier.model.util import get_class_weights
from fake_news_classifier.util import get_tb_logdir, log

# Arguments: Init
SEQ_LEN = 'seq_len'
EMB_DIM = 'emb_dim'
DROPOUT = 'dropout'
LSTM_UNITS = 'lstm_units'
DENSE_UNITS = 'dense_units'
# Arguments: Train/Predict
VAL_SPLIT = 'val_split'
SAVE_LOGS = 'save_logs'
NUM_EPOCHS = 'num_epochs'
VERBOSE = 'verbose'
BATCH_SIZE = 'batch_size'


# Input LSTM unit - create one for each of the text inputs
def get_input_lstm(input_shape, dropout, num_units):
    input_lstm = Sequential()
    lstm_1 = Bidirectional(
        LSTM(units=num_units, dropout=dropout, recurrent_dropout=dropout),
        input_shape=input_shape
    )
    input_lstm.add(lstm_1)
    return input_lstm


class BiLSTMWithDense(FNCModel):
    """
    Two bidirectional LSTM input neurons -> Concatenation -> Normalization -> 2 Dense -> Output Dense -> Softmax
    Arguments (Init):
        - Sequence Length (required)
        - Embedding dimension (required)
        - Dropout: 0-1 (used for dropout, recurrent_dropout for LSTM) - default 0.5
        - LSTM Units - default 128
        - Dense Units - default 64

    Arguments (Train):
        - Validation split - default 0.2
        - Whether to save logs - default False
        - Number of epochs - default 25
        - Verbosity - default 2 (one line per epoch)
        - Batch size - default 32

    Arguments (Predict):
        - Verbosity - default 1
        - Batch size - default 32

    # TODO: Things to try:
        - Regularization
        - Dropout
        - Depth/Width
        - Learning rate
    """

    def __init__(self, args, name='BiLSTMWithDense'):
        super(BiLSTMWithDense, self).__init__(name, args)

        # Get args for building the model, default to some accepted parameters
        seq_len = self.args.get(SEQ_LEN)
        emb_dim = self.args.get(EMB_DIM)
        dropout = self.args.get(DROPOUT, 0.5)
        lstm_num_units = self.args.get(LSTM_UNITS, 128)
        dense_num_hidden = self.args.get(DENSE_UNITS, 64)

        input_shape = (seq_len, emb_dim)

        text_one_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units
        )
        text_two_lstm = get_input_lstm(
            input_shape=input_shape,
            dropout=dropout,
            num_units=lstm_num_units
        )

        merged_mlp = Concatenate()([text_one_lstm.output, text_two_lstm.output])
        merged_mlp = BatchNormalization()(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dense(3, activation='softmax')(merged_mlp)

        complete_model = Model([text_one_lstm.input, text_two_lstm.input], merged_mlp)

        # Create the optimizer
        optimizer = Adam()

        complete_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        complete_model.summary()
        # Instance variables
        self.model = complete_model
        self.seq_len = seq_len
        self.log_name = f"{self.name}_{lstm_num_units}L-{dense_num_hidden}D-{dropout}drop"

    # Model Training
    # Data should be a pandas dataframe (or dict) with the indicies defined within this class
    # Data should be vectorized if text, labels should be either 0,1, or 2
    def train(self, data, train_args):
        # Args
        save_logs = train_args.get(SAVE_LOGS, False)
        val_split = train_args.get(VAL_SPLIT, 0.2)
        batch_size = train_args.get(BATCH_SIZE, 32)
        epochs = train_args.get(NUM_EPOCHS, 25)
        verbose = train_args.get(VERBOSE, 2)
        # Input data
        texts = data[TEXT_ONE_IDX]
        other_texts = data[TEXT_TWO_IDX]
        labels = data[LABEL_IDX]

        # Do sequence padding
        texts = pad_sequences(texts, maxlen=self.seq_len, dtype='float32')
        other_texts = pad_sequences(other_texts, maxlen=self.seq_len, dtype='float32')
        labels = to_categorical(labels, num_classes=3)
        class_weights = get_class_weights(labels)
        log("Calculated class weights")
        log(class_weights)
        # Init tensorboard
        callbacks = []
        if save_logs:
            callbacks.append(TensorBoard(log_dir=get_tb_logdir(self.log_name)))
        return self.model.fit(
            [texts, other_texts],
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=val_split,
            class_weight=class_weights,
            callbacks=callbacks
        )

    # Use model to predict
    def predict(self, data, predict_args):
        # Get args
        batch_size = predict_args.get(BATCH_SIZE, 32)
        verbose = predict_args.get(VERBOSE, 1)
        # Get data
        texts = data[TEXT_ONE_IDX]
        other_texts = data[TEXT_TWO_IDX]

        titles = pad_sequences(texts, maxlen=self.seq_len, dtype='float32')
        bodies = pad_sequences(other_texts, maxlen=self.seq_len, dtype='float32')
        return self.model.predict(
            [titles, bodies],
            batch_size=batch_size,
            verbose=verbose
        )

    # Save model to disk
    def save(self, path):
        pass
