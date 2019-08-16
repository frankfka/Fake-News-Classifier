from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

from fake_news_classifier.preprocessing.text_util import tokenize_by_word, clean_tokenized, analyze_pos,\
    combine_token_pos
from fake_news_classifier.model.FNCModel import FNCModel


def tokenizer(txt):
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
    return tokenized


def get_nn(input_dim):
    nn = Sequential()
    nn.add(Dense(512, input_dim=input_dim, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(Dense(256, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(Dense(1, activation='sigmoid'))
    return nn


class ArticleCredibilityNN(FNCModel):
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
        super(ArticleCredibilityNN, self).__init__(name, args)
        # Get arguments
        tfidf_max_df = args.get(ArticleCredibilityNN.TFIDF_MAX_DF, 0.99)
        tfidf_min_df = args.get(ArticleCredibilityNN.TFIDF_MIN_DF, 0.01)

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            max_df=tfidf_max_df,
            min_df=tfidf_min_df,
            ngram_range=(1, 2),
            tokenizer=tokenizer
        )

    # Model Training - Dataframe with TEXT_IDX and LABEL_IDX as columns
    #   TEXT_IDX column should be uncleaned, raw string
    #   LABEL_IDX column should be 0 for false, 1 for true
    def train(self, data, train_args):
        texts = data[ArticleCredibilityNN.TEXT_IDX]
        labels = data[ArticleCredibilityNN.LABEL_IDX]
        vec_texts = self.vectorizer.fit_transform(texts)
        input_dim = vec_texts.shape[1]
        nn = get_nn(input_dim=input_dim)
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = nn
        return self.model.fit(
            x=vec_texts,
            y=labels,
            epochs=30,
            batch_size=64,
            validation_split=0.2
        )

    # Use model to predict - Data should just be a list of strings (articles)
    def predict(self, data, predict_args):
        texts = self.vectorizer.transform(data)
        return self.model.predict(texts)

    # Save model to disk
    def save(self, path):
        pass


if __name__ == '__main__':

    import pandas as pd
    raw_df = pd.read_pickle('../data/other/joined_articles_dataset.pkl')
    raw_df = raw_df.loc[raw_df['text'].str.len() > 500]
    raw_df = raw_df
    from sklearn.model_selection import train_test_split
    raw_train, raw_test = train_test_split(raw_df, test_size=0.20, random_state=42)

    X_train = raw_train['text']
    Y_train = raw_train['label']

    X_test = raw_test['text']
    Y_test_true = raw_test['label']

    model = ArticleCredibilityNN(args={})

    train_df = pd.DataFrame(
        data={
            ArticleCredibilityNN.TEXT_IDX: X_train,
            ArticleCredibilityNN.LABEL_IDX: Y_train
        }
    )
    model.train(train_df, train_args={})
    from fake_news_classifier.model.util import eval_predictions, categorical_to_idx
    Y_test_pred = model.predict(X_test, predict_args={})
    Y_test_pred = categorical_to_idx(Y_test_pred)

    eval_predictions(Y_test_true, Y_test_pred, classes=['fake', 'real'], print_results=True)

    model.save('./test.pkl')
