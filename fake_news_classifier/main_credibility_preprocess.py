import time

import pandas as pd

from fake_news_classifier.const import LABEL_IDX, CRED_IDX, TEXT_TWO_IDX, TEXT_ONE_IDX
from fake_news_classifier.model.ArticleCredibilityPAC import ArticleCredibilityPAC
from fake_news_classifier.preprocessing.GensimVectorizer import GensimVectorizer
from fake_news_classifier.preprocessing.preprocess_nn import preprocess_nn


# Load Data (json_data, articles_data)
def load_raw_data(json_pkl_path, articles_pkl_path):
    return pd.read_pickle(json_pkl_path), pd.read_pickle(articles_pkl_path)


v = GensimVectorizer(
    path='./preprocessing/assets/300d.commoncrawl.fasttext.vec',
    binary=False
)
credibility_pac = ArticleCredibilityPAC(args={
    ArticleCredibilityPAC.PKL_PATH: './model/ArticleCredibilityPAC_Trained.pkl'
})

json_df, articles_df = load_raw_data('./data/fnc_json_data.pkl', './data/fnc_articles_data.pkl')
texts, other_texts, labels, credibilities = preprocess_nn(
    json_df,
    articles_df,
    v,
    max_seq_len=256,
    credibility_model=credibility_pac
)

df = pd.DataFrame(data={
    TEXT_ONE_IDX: texts,
    TEXT_TWO_IDX: other_texts,
    CRED_IDX: credibilities,
    LABEL_IDX: labels
})

print(df.head())
df.to_pickle('./data/fnc_train_data_individual.pkl')