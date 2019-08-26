from keras import Sequential, Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Bidirectional, LSTM, BatchNormalization, Concatenate, Dense, Dropout, MaxPooling1D, Conv1D
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
CONV_UNITS = 'conv_units'
CONV_KERNEL_SIZE = 'conv_kernel_size'
DENSE_UNITS = 'dense_units'
LEARN_RATE = 'learn_rate'
# Arguments: Train/Predict
VAL_SPLIT = 'val_split'
SAVE_LOGS = 'save_logs'
EARLY_STOP = 'early_stop'
NUM_EPOCHS = 'num_epochs'
VERBOSE = 'verbose'
BATCH_SIZE = 'batch_size'


# Input C-LSTM unit - create one for each of the text inputs
# Conv1D -> Dropout -> MaxPool -> Bi-LSTM
def get_input_nn(input_shape, dropout, num_lstm_units, num_conv_units, conv_kernel_size):
    nn = Sequential()
    nn.add(
        Conv1D(
            filters=num_conv_units,
            kernel_size=conv_kernel_size,
            activation='relu',
            input_shape=input_shape
        )
    )
    nn.add(Dropout(dropout))
    nn.add(MaxPooling1D())
    nn.add(
        Bidirectional(
            LSTM(units=num_lstm_units, dropout=dropout, recurrent_dropout=dropout)
        )
    )
    return nn


class CLSTMWithDense(FNCModel):
    """
    Two Inputs (CNN -> Dropout -> MaxPool -> LSTM)
     -> Concatenation -> Normalization -> 2 Dense -> Output Dense -> Softmax
    Arguments (Init):
        - Sequence Length (required)
        - Embedding dimension (required)
        - Dropout: 0-1 (used for dropout, recurrent_dropout for LSTM) - default 0.5
        - LSTM Units - default 128
        - Conv Units - default 256
        - Conv Pool Size - default 3
        - Dense Units - default 64

    Arguments (Train):
        - Validation split - default 0.2
        - Whether to save logs - default False
        - Whether to stop early when validation loss stops decreasing - default False
        - Number of epochs - default 25
        - Verbosity - default 2 (one line per epoch)
        - Batch size - default 32

    Arguments (Predict):
        - Verbosity - default 1
        - Batch size - default 32
    """

    def __init__(self, args, name='CLSTMWithDense'):
        super(CLSTMWithDense, self).__init__(name, args)

        # Get args for building the model, default to some accepted parameters
        seq_len = self.args.get(SEQ_LEN)
        emb_dim = self.args.get(EMB_DIM)
        dropout = self.args.get(DROPOUT, 0.5)
        conv_num_units = self.args.get(CONV_UNITS, 256)
        conv_kernel_size = self.args.get(CONV_KERNEL_SIZE, 3)
        lstm_num_units = self.args.get(LSTM_UNITS, 128)
        dense_num_hidden = self.args.get(DENSE_UNITS, 64)
        learn_rate = self.args.get(LEARN_RATE, 0.001)

        input_shape = (seq_len, emb_dim)

        text_one_nn = get_input_nn(
            input_shape=input_shape,
            dropout=dropout,
            num_lstm_units=lstm_num_units,
            num_conv_units=conv_num_units,
            conv_kernel_size=conv_kernel_size
        )
        text_two_nn = get_input_nn(
            input_shape=input_shape,
            dropout=dropout,
            num_lstm_units=lstm_num_units,
            num_conv_units=conv_num_units,
            conv_kernel_size=conv_kernel_size
        )

        merged_mlp = Concatenate()([text_one_nn.output, text_two_nn.output])
        merged_mlp = BatchNormalization()(merged_mlp)
        merged_mlp = Dropout(dropout)(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dropout(dropout)(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dropout(dropout)(merged_mlp)
        merged_mlp = Dense(3, activation='softmax')(merged_mlp)

        complete_model = Model([text_one_nn.input, text_two_nn.input], merged_mlp)

        # Create the optimizer
        optimizer = Adam(lr=learn_rate)

        complete_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        complete_model.summary()
        # Instance variables
        self.model = complete_model
        self.seq_len = seq_len
        self.log_name = f"{self.name}_{conv_num_units}C_{lstm_num_units}L-{dense_num_hidden}D-{dropout}drop"

    # Model Training
    # Data should be a pandas dataframe (or dict) with the indicies defined within this class
    # Data should be vectorized if text, labels should be either 0,1, or 2
    def train(self, data, train_args):
        # Args
        save_logs = train_args.get(SAVE_LOGS, False)
        early_stop = train_args.get(EARLY_STOP, False)
        val_split = train_args.get(VAL_SPLIT, 0.2)
        batch_size = train_args.get(BATCH_SIZE, 64)
        epochs = train_args.get(NUM_EPOCHS, 25)
        verbose = train_args.get(VERBOSE, 2)
        # Input data
        texts = data[TEXT_ONE_IDX]
        other_texts = data[TEXT_TWO_IDX]
        labels = data[LABEL_IDX]

        # Do sequence padding
        texts = pad_sequences(texts, maxlen=self.seq_len, dtype='float16', truncating='post')
        other_texts = pad_sequences(other_texts, maxlen=self.seq_len, dtype='float16', truncating='post')
        labels = to_categorical(labels, num_classes=3)
        class_weights = get_class_weights(labels)
        log("Calculated class weights")
        log(class_weights)
        # Init tensorboard
        callbacks = []
        if save_logs:
            callbacks.append(TensorBoard(log_dir=get_tb_logdir(self.log_name)))
        if early_stop:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, min_delta=0.003))
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

        titles = pad_sequences(texts, maxlen=self.seq_len, dtype='float16', truncating='post')
        bodies = pad_sequences(other_texts, maxlen=self.seq_len, dtype='float16', truncating='post')
        return self.model.predict(
            [titles, bodies],
            batch_size=batch_size,
            verbose=verbose
        )

    # Save model to disk
    def save(self, path):
        pass
