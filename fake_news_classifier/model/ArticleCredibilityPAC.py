from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

from fake_news_classifier.preprocessing.text_util import tokenize_by_word, clean_tokenized, analyze_pos,\
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


class ArticleCredibilityPAC(FNCModel):
    """
    Passive-Aggressive classifier to identify article credibility
    Arguments (Init):
        - Pickle path - will load model from pickle if specified
        - TFIDF max_df - default 0.99
        - TFIDF min_df - default 0.01
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

    def __init__(self, args, name='ArticleCredibilityPAC'):
        super(ArticleCredibilityPAC, self).__init__(name, args)
        # Load from pickle if specified
        pickle_path = args.get(ArticleCredibilityPAC.PKL_PATH, None)
        if pickle_path is not None:
            self.model, self.vectorizer = load_from_pickle(pickle_path)
            return
        # Get arguments
        tfidf_max_df = args.get(ArticleCredibilityPAC.TFIDF_MAX_DF, 0.99)
        tfidf_min_df = args.get(ArticleCredibilityPAC.TFIDF_MIN_DF, 0.01)

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            max_df=tfidf_max_df,
            min_df=tfidf_min_df,
            ngram_range=(1, 2)
        )
        self.model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3)

    # Model Training - Dataframe with TEXT_IDX and LABEL_IDX as columns
    #   TEXT_IDX column should be uncleaned, raw string
    #   LABEL_IDX column should be 0 for false, 1 for true
    def train(self, data, train_args):
        texts = data[ArticleCredibilityPAC.TEXT_IDX]
        texts = [preprocess(txt) for txt in texts]
        labels = data[ArticleCredibilityPAC.LABEL_IDX]
        vec_texts = self.vectorizer.fit_transform(texts)
        self.model.fit(vec_texts, labels)

    # Use model to predict - Data should just be a list of strings (articles)
    def predict(self, data, predict_args):
        texts = [preprocess(txt) for txt in data]
        texts = self.vectorizer.transform(texts)
        return self.model.predict(texts)

    # Save model to disk
    def save(self, path):
        dict_to_save = {
            ArticleCredibilityPAC.PKL_PAC_KEY: self.model,
            ArticleCredibilityPAC.PKL_TFIDF_KEY: self.vectorizer
        }
        with open(path, 'wb') as file:
            pickle.dump(dict_to_save, file)


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

    model = ArticleCredibilityPAC(args={})

    train_df = pd.DataFrame(
        data={
            ArticleCredibilityPAC.TEXT_IDX: X_train,
            ArticleCredibilityPAC.LABEL_IDX: Y_train
        }
    )
    model.train(train_df, train_args={})
    Y_test_pred = model.predict(X_test, predict_args={})
    from fake_news_classifier.model.util import eval_predictions
    eval_predictions(Y_test_true, Y_test_pred, classes=['fake', 'real'], print_results=True)

    model.save('./ArticleCredibilityPAC_Trained.pkl')
