from keras import Model, Input
from keras.backend import mean
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Bidirectional, LSTM, Concatenate, Dense, Dropout, \
    RepeatVector, Lambda, Dot, Activation
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
LEARN_RATE = 'learn_rate'
# Arguments: Train/Predict
VAL_SPLIT = 'val_split'
SAVE_LOGS = 'save_logs'
EARLY_STOP = 'early_stop'
NUM_EPOCHS = 'num_epochs'
VERBOSE = 'verbose'
BATCH_SIZE = 'batch_size'


class DeclareLSTM(FNCModel):
    """
    Based on the Declare model
    Arguments (Init):
        - Sequence Length (required)
        - Embedding dimension (required)
        - Dropout: 0-1 (used for dropout, recurrent_dropout for LSTM) - default 0.5
        - LSTM Units - default 64
        - Dense Units - default 32
        - Learning Rate - default 0.002

    Arguments (Train):
        - Validation split - default 0.2
        - Whether to save logs - default False
        - Whether to stop early when validation loss stops decreasing - default False
        - Number of epochs - default 25
        - Verbosity - default 2 (one line per epoch)
        - Batch size - default 64

    Arguments (Predict):
        - Verbosity - default 2
        - Batch size - default 64
    """

    def __init__(self, args, name='DeclareLSTM'):
        super(DeclareLSTM, self).__init__(name, args)

        # Get args for building the model, default to some accepted parameters
        seq_len = self.args.get(SEQ_LEN)
        emb_dim = self.args.get(EMB_DIM)
        dropout = self.args.get(DROPOUT, 0.5)
        lstm_num_units = self.args.get(LSTM_UNITS, 64)
        dense_num_hidden = self.args.get(DENSE_UNITS, 32)
        learn_rate = self.args.get(LEARN_RATE, 0.002)

        input_shape = (seq_len, emb_dim)

        txt_one_input = Input(shape=input_shape)
        txt_two_input = Input(shape=input_shape)

        '''
        One side of the network
        - Take the mean of the input vectors, repeat it by seq_len
        - Concatenate with the article vectors
        - Pass through a dense NN and an activation
        '''
        mean_layer = Lambda(lambda x: mean(x, axis=-2))(txt_one_input)
        txt_one_mean = RepeatVector(seq_len)(mean_layer)
        attn_input = Concatenate()([txt_one_mean, txt_two_input])
        attn_nn = Dense(lstm_num_units * 2, activation='relu')(attn_input)
        attn_nn = Activation('softmax')(attn_nn)
        attn_model = Model([txt_one_input, txt_two_input], attn_nn)

        '''
        Other side of the network
        - Pass article through a bidirectional LSTM
        '''
        lstm = Bidirectional(LSTM(lstm_num_units, return_sequences=True), merge_mode='concat')(txt_two_input)
        lstm_model = Model([txt_two_input], lstm)

        '''
        Final conjoined network
        - Take dot product of attention model and article LSTM
        - Take an average cross one dimension to get a 2D tensor
        - Pass through dense layers then a prediction layer
        '''
        inner_pdt = Dot(axes=1)([attn_model.output, lstm_model.output])
        avg = Lambda(lambda x: mean(x, axis=-1))(inner_pdt)

        merged_mlp = Dense(dense_num_hidden * 2, activation='relu')(avg)
        merged_mlp = Dropout(dropout)(merged_mlp)
        merged_mlp = Dense(dense_num_hidden, activation='relu')(merged_mlp)
        merged_mlp = Dropout(dropout)(merged_mlp)
        merged_mlp = Dense(3, activation='softmax')(merged_mlp)

        complete_model = Model([txt_one_input, txt_two_input], merged_mlp)

        # Create the optimizer
        optimizer = Adam(lr=learn_rate)

        complete_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        complete_model.summary()
        self.model = complete_model
        self.seq_len = seq_len
        self.log_name = f"{self.name}_{lstm_num_units}L-{dense_num_hidden}D-{dropout}drop"

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
        batch_size = predict_args.get(BATCH_SIZE, 64)
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