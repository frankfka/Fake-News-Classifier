import pickle

from RMDL.RMDL_Text import Text_Classification

from fake_news_classifier.preprocessing.text_util import tokenize_by_word, clean_tokenized, analyze_pos, \
    combine_token_pos
from fake_news_classifier.model.FNCModel import FNCModel


# Load model from disk
def load_from_pickle(path):
    with open(path, 'rb') as file:
        loaded_dict = pickle.load(file)
    loaded_model = loaded_dict[ArticleCredibilityPAC.PKL_PAC_KEY]
    loaded_vec = loaded_dict[ArticleCredibilityPAC.PKL_TFIDF_KEY]
    return loaded_model, loaded_vec


# Called before training and predicting on a specific corpus
def preprocess(txt):
    # Tokenize by word
    tokenized = tokenize_by_word(txt)
    # Clean the tokenized
    tokenized = clean_tokenized(
        tokenized,
        lowercase=True,
        remove_stopwords=True,
        remove_punctuation=True
    )
    # Analyze part of speech, lemmatize if needed
    tokenized = analyze_pos(tokenized, lemmatize=True)
    # (word, pos) -> word_pos
    tokenized = combine_token_pos(tokenized)
    return ' '.join(tokenized)


class ArticleCredibilityRMDL(FNCModel):
    """
    RMDL Model
    """

    # Input dataframe indicies
    TEXT_IDX = 'text'
    LABEL_IDX = 'label'
    # Keys to retrieve from pickled object
    PKL_PAC_KEY = 'pac'
    PKL_TFIDF_KEY = 'tfidf'
    # Model init params
    TFIDF_MAX_DF = 'tfidf_max_df'
    TFIDF_MIN_DF = 'tfidf_min_df'
    PKL_PATH = 'pkl_path'

    def __init__(self, args, name='ArticleCredibilityRMDL'):
        super(ArticleCredibilityRMDL, self).__init__(name, args)
        # TODO
        self.vectorizer = None
        self.model = None

    # Model Training - Dataframe with TEXT_IDX and LABEL_IDX as columns
    #   TEXT_IDX column should be uncleaned, raw string
    #   LABEL_IDX column should be 0 for false, 1 for true
    def train(self, data, train_args):
        # TODO
        texts = data[ArticleCredibilityRMDL.TEXT_IDX]
        texts = [preprocess(txt) for txt in texts]
        labels = data[ArticleCredibilityRMDL.LABEL_IDX]
        vec_texts = self.vectorizer.fit_transform(texts)
        self.model.fit(vec_texts, labels)

    # Use model to predict - Data should just be a list of strings (articles)
    def predict(self, data, predict_args):
        # TODO
        texts = [preprocess(txt) for txt in data]
        texts = self.vectorizer.transform(texts)
        return self.model.predict(texts)

    # Save model to disk
    def save(self, path):
        pass


if __name__ == '__main__':
    import pandas as pd

    raw_df = pd.read_pickle('../data/other/joined_articles_dataset.pkl')
    raw_df = raw_df.loc[raw_df['text'].str.len() > 500]
    from sklearn.model_selection import train_test_split

    raw_train, raw_test = train_test_split(raw_df, test_size=0.20, random_state=42)

    X_train = raw_train['text']
    Y_train = raw_train['label']

    X_test = raw_test['text']
    Y_test_true = raw_test['label']

    Text_Classification(X_train, Y_train, X_test, Y_test_true, batch_size=128,
                        EMBEDDING_DIM=300, MAX_SEQUENCE_LENGTH=500, MAX_NB_WORDS=75000,
                        GloVe_dir="../preprocessing/assets", GloVe_file="300d.commoncrawl.glove.txt",
                        sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=True,
                        min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                        min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32, max_nodes_rnn=128,
                        min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                        random_state=42, random_optimizor=True, dropout=0.4)
