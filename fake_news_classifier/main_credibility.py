import time

import pandas as pd

import fake_news_classifier.model.CredCLSTMWithDense as model
from fake_news_classifier.const import LABEL_IDX, CRED_IDX, TEXT_TWO_IDX, TEXT_ONE_IDX, CLAIM_ID_IDX
from fake_news_classifier.model.ArticleCredibilityPAC import ArticleCredibilityPAC
from fake_news_classifier.model.util import categorical_to_idx, plot_keras_history, eval_predictions
from fake_news_classifier.preprocessing.FNCData import FNCData
from fake_news_classifier.preprocessing.GensimVectorizer import GensimVectorizer
from fake_news_classifier.preprocessing.preprocess_nn import preprocess_nn


# Load Data (json_data, articles_data)
from fake_news_classifier.util import log


def load_raw_data(json_pkl_path, articles_pkl_path):
    return pd.read_pickle(json_pkl_path), pd.read_pickle(articles_pkl_path)


# Preprocess, returns (texts, other_texts, labels, credibilities) in a FNCData object, pickles if pkl is set
def preprocess(json_data, articles_data, credibility_model, vectorizer, max_seq_len, max_label_bias=None, pkl=None):
    texts, other_texts, labels, creds = preprocess_nn(
        json_data,
        articles_data,
        vectorizer,
        max_seq_len=256,
        credibility_model=credibility_model
    )
    data = FNCData(texts, other_texts, labels, vectorizer, max_seq_len, max_label_bias, creds)
    if pkl is not None:
        data.data.to_pickle(pkl)
    return data


# Load preprocessed data, returns FNCData object
def load_preprocessed(pkl_path, vectorizer, max_seq_len, max_label_bias=None, fnc_pkl_path=None):
    df = pd.read_pickle(pkl_path)
    if fnc_pkl_path is not None:
        df_fnc = pd.read_pickle(fnc_pkl_path)
        df = pd.concat([df, df_fnc], ignore_index=True)
    return FNCData(
        list_of_txt=df[TEXT_ONE_IDX],
        other_list_of_txt=df[TEXT_TWO_IDX],
        list_of_labels=df[LABEL_IDX],
        vectorizer=vectorizer,
        max_seq_len=max_seq_len,
        max_label_bias=max_label_bias,
        list_of_cred=df[CRED_IDX]
    )


# Returns a vectorized dataframe input to the model, given an FNCData object
def load_batch(fnc_data, idx=None):
    vec_txt, vec_other_txt, creds, labels = fnc_data.get(vectorize=True, idx=idx, use_ngrams=True)
    return pd.DataFrame(data={
        TEXT_ONE_IDX: vec_txt,
        TEXT_TWO_IDX: vec_other_txt,
        CRED_IDX: creds,
        LABEL_IDX: labels
    })


# Build, Train, and Evaluate Model - Returns incorrect (idx, prediction) from test set
def build_train_eval(train_df, test_df):
    # Build Model
    model_args = {
        model.SEQ_LEN: 500,
        model.EMB_DIM: 300,
        model.CONV_KERNEL_SIZE: 2,
        model.DENSE_UNITS: 1024,
        model.CONV_UNITS: 256,
        model.LSTM_UNITS: 128
    }
    nn = model.CredCLSTMWithDense(model_args)
    # Train model
    train_args = {
        model.BATCH_SIZE: 128,
        model.NUM_EPOCHS: 30,
        model.EARLY_STOP: True
    }
    history = nn.train(data=train_df, train_args=train_args)

    # Evaluate
    y_val_true = test_df[LABEL_IDX]
    y_val_pred = nn.predict(test_df, predict_args={})
    y_val_pred = categorical_to_idx(y_val_pred)  # Has claims

    plot_keras_history(history, True)

    log("Evaluating Raw Results", header=True)
    eval_predictions(y_true=y_val_true, y_pred=y_val_pred,
                     classes=['disagree (0)', 'discuss (1)', 'agree (2)'], print_results=True)

    log("Evaluating Processed Results", header=True)

    # TODO: Temporary solution to incorporate claim ID to verify prediction
    # Save first so we don't lose all the information
    pd.DataFrame(
        data={'claim': test_df[TEXT_ONE_IDX], 'pred': y_val_pred, 'true': y_val_true}
    ).to_pickle('./raw_pred_true.pkl')
    log("Saved raw predictions")

    claim_ids = list(pd.read_pickle('./data/processed/test_data_individual_claimid_credible.pkl')['claim_id'])

    # Key is claim ID, value is list of predictions
    pred_dict = dict()
    true_dict = dict()
    for idx, claim_id in enumerate(claim_ids):
        pred = y_val_pred[idx]
        true = y_val_true[idx]
        if claim_id in pred_dict:
            pred_dict[claim_id].append(pred)
            true_dict[claim_id].append(true)
        else:
            pred_dict[claim_id] = [pred]
            true_dict[claim_id] = [true]

    # Get keys
    dict_keys = list(pred_dict.keys())

    # Iterate over keys, get mean of lists
    import numpy as np

    true = []
    pred = []
    for key in dict_keys:
        true.append(
            int(np.mean(true_dict[key]))
        )
        pred.append(
            int(round(float(np.mean(pred_dict[key]))))
        )

    # Save
    processed_results_df = pd.DataFrame(data={
        'claim_id': dict_keys,
        'pred': pred,
        'true_label': true
    })
    processed_results_df.to_pickle('./processed_pred_true_id_final.pkl')
    eval_predictions(y_true=true, y_pred=pred,
                     classes=['disagree (0)', 'discuss (1)', 'agree (2)'], print_results=True)


# Load all dependencies
checkpoint_time = time.time()
log("Loading Preprocessed Data", header=True)
v = GensimVectorizer(
    path='./preprocessing/assets/300d.commoncrawl.fasttext.vec',
    binary=False
)
credibility_pac = ArticleCredibilityPAC(args={
    ArticleCredibilityPAC.PKL_PATH: './model/ArticleCredibilityPAC_Trained.pkl'
})

data = load_preprocessed(
    pkl_path='./data/processed/train_data_individual_claimid_nodup.pkl',
    # fnc_pkl_path='./data/train_data_fnc.pkl',
    vectorizer=v,
    max_seq_len=256,
    max_label_bias=1.5
)
test_data = load_preprocessed(
    pkl_path='./data/processed/test_data_individual_claimid_credible.pkl',
    vectorizer=v,
    max_seq_len=256
)
# json_df, articles_df = load_raw_data('./data/json_data.pkl', './data/articles_data.pkl')
# data = preprocess(json_df, articles_df, vectorizer=v, max_seq_len=500)
# data.data.to_pickle('./data/train_data_individual.pkl')

now = time.time()
log(f"Loaded preprocessed data in {now - checkpoint_time}s")
checkpoint_time = now

log("Training", header=True)

trainable_df = load_batch(data)
testable_df = load_batch(test_data)

# Train, eval model, then get the failed indicies and save them for processing
build_train_eval(train_df=trainable_df, test_df=testable_df)

now = time.time()
log(f"Training completed in {now - checkpoint_time} seconds")