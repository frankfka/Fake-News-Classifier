import time

import pandas as pd

import fake_news_classifier.model.BiLSTMWithDense as model
import fake_news_classifier.const as const
from fake_news_classifier.model.util import plot_keras_history, categorical_to_idx, eval_predictions, k_fold_indicies
from fake_news_classifier.preprocessing.FNCData import FNCData
from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer
from fake_news_classifier.preprocessing.preprocess_nn import preprocess_nn
from fake_news_classifier.util import log


# Load Data (json_data, articles_data)
def load_raw_data(json_pkl_path, articles_pkl_path):
    return pd.read_pickle(json_pkl_path), pd.read_pickle(articles_pkl_path)


# Preprocess data, returns FNCData object
def preprocess(json_data, articles_data, vectorizer, max_seq_len):
    texts, other_texts, labels = preprocess_nn(json_data, articles_data, vectorizer, max_seq_len=max_seq_len)
    return FNCData(
        list_of_txt=texts,
        other_list_of_txt=other_texts,
        list_of_labels=labels,
        vectorizer=vectorizer,
        max_seq_len=500
    )


# Load preprocessed data, returns FNCData object
def load_preprocessed(pkl_path, vectorizer, max_seq_len):
    df = pd.read_pickle(pkl_path)
    return FNCData(
        list_of_txt=df[const.TEXT_ONE_IDX],
        other_list_of_txt=df[const.TEXT_TWO_IDX],
        list_of_labels=df[const.LABEL_IDX],
        vectorizer=vectorizer,
        max_seq_len=max_seq_len
    )


# Gets indicies for k-fold validation from FNCData
def k_fold(fnc_data, k):
    x, x_other, y = fnc_data.get()
    return k_fold_indicies(x, y, k)


# Returns a vectorized dataframe input to the model, given an FNCData object
def load_batch(fnc_data, idx):
    vec_txt, vec_other_txt, labels = fnc_data.get(vectorize=True, idx=idx)
    return pd.DataFrame(data={
        const.TEXT_ONE_IDX: vec_txt,
        const.TEXT_TWO_IDX: vec_other_txt,
        const.LABEL_IDX: labels
    })


# Build, Train, and Evaluate Model
def build_train_eval(train_df, test_df):
    # Build Model
    model_args = {
        model.SEQ_LEN: 500,
        model.EMB_DIM: 300
    }
    bi_lstm = model.BiLSTMWithDense(model_args)
    # Train model
    history = bi_lstm.train(data=train_df, train_args={})

    # Evaluate
    y_val_true = test_df[const.LABEL_IDX]
    y_val_pred = bi_lstm.predict(test_df, predict_args={})
    y_val_pred = categorical_to_idx(y_val_pred)
    plot_keras_history(history, True)
    eval_predictions(y_true=y_val_true, y_pred=y_val_pred, print_results=True)


checkpoint_time = time.time()
log("Loading Preprocessed Data", header=True)

v = GoogleNewsVectorizer()
data = load_preprocessed(
    pkl_path='./data/train_data_all.pkl',
    vectorizer=v,
    max_seq_len=500
)

now = time.time()
log(f"Loaded preprocessed data in {now - checkpoint_time}s")
checkpoint_time = now

# Train with k-fold validation
for fold, (train_idx, test_idx) in enumerate(k_fold(data, k=5)):

    log(f"Training Fold {fold}", header=True)

    train_data = load_batch(data, train_idx)
    test_data = load_batch(data, test_idx)

    build_train_eval(train_df=train_data, test_df=test_data)

    now = time.time()
    log(f"Fold {fold} completed in {now - checkpoint_time} seconds")
    checkpoint_time = now
