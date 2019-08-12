import re
import time

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import fake_news_classifier.const as const
from fake_news_classifier.preprocessing.text_util import tokenize_by_sent, tokenize_by_word
from fake_news_classifier.util import log

"""
This class abstracts away the details of the FNC dataset. This is so that our models can be generalized.
If the dataset changes, only the preprocessors need to change.
"""


def preprocess_nn(json_df, articles_df, vectorizer, max_seq_len):
    """
    Given the raw FNC data, return 3 lists of (text, other_text (supporting info), and labels)
    """
    claims = json_df[const.PKL_CLAIM]
    labels = json_df[const.PKL_LABEL]
    related_articles = json_df[const.PKL_RELATED_ARTICLES]
    supporting_info = []  # Stores the processed supporting information for each claim

    start_time = time.time()  # Used for tracking only

    for j, (str_claim, article_ids) in enumerate(zip(claims, related_articles)):

        # Tracking
        if j % 1000 == 0 and j != 0:
            now = time.time()
            log(f"Processing claim {j} | Last 1000 claims took {now - start_time} seconds")
            start_time = now

        # Get list of article bodies from the dataframe
        article_ids = [str(article_id) for article_id in article_ids]  # Need to lookup by string
        # Get the articles with the given article ID's and only extract the text column
        articles = articles_df.loc[articles_df[const.PKL_ARTICLE_ID].isin(article_ids), const.PKL_ARTICLE_TXT]
        support_txt = get_relevant_info(str_claim, articles, vectorizer, max_seq_len)
        supporting_info.append(support_txt)

    return claims, supporting_info, labels


def get_relevant_info(claim, articles, vectorizer, max_seq_len):
    """
    Returns the most relevant sentences relating to a claim using the average vectors of words and cosine similarity
    - Extra whitespace is trimmed and removed
    - Punctuation is removed
    - Maintains upper/lowercase


    - TODO: Process long sentences by splitting them up
    - TODO: Similarity by checking for named entities, numbers
    - TODO: Additional text cleaning, like lemmatization? Remove stopwords?
    """
    vec_claim = vectorizer.transform_one(claim)  # Claim vector - we'll use this to compare using cosine similarity
    similarities_and_sents = []  # Stores tuples of (cos sim, sentence)

    for article in articles:
        '''
        For each article, we split it into sentences
            For each sentence, we clean and vectorize, then retrieve the cosine similarity of the claim vs the sentence
            - Remove extra whitespace
            - Skip very short sentences
            - Other potential cleaning
        '''
        sentences = tokenize_by_sent(article)
        for sentence in sentences:
            sentence = re.sub(r'\W+', ' ', sentence).strip()  # Remove extra whitespace
            # Don't process for sentences less than 20 characters long - this usually means improper sentences/words
            if len(sentence) < 20:
                continue
            vec_sent = vectorizer.transform_one(sentence)
            similarity = cos_sim(vec_claim, vec_sent)
            similarities_and_sents.append((similarity, sentence))

    # Sort the similarities (in desc order) using their similarity
    sorted_sents = sorted(similarities_and_sents, key=lambda elem: elem[0], reverse=True)

    # Construct relevant info - keep looping through sentences, adding word by word until we hit max_seq_len
    article_info = ''
    num_words = 0
    for similarity, sentence in sorted_sents:
        if num_words >= max_seq_len:
            break
        words = tokenize_by_word(sentence)
        for word in words:
            article_info += ' ' + word
            num_words += 1
            if num_words >= max_seq_len:
                break
    return article_info


# Returns cosine similarity between two texts
def cos_sim(vec_text, other_vec_text):
    if len(vec_text) == 0 or len(other_vec_text) == 0:
        return 0
    # Get average vectors of the texts
    avg_vec = get_avg_vec(vec_text).reshape(1, -1)  # Reshape to get a 2D array of a single sample
    avg_other_vec = get_avg_vec(other_vec_text).reshape(1, -1)
    cos_similarities = cosine_similarity(avg_vec, avg_other_vec)  # n samples x n samples matrix
    return np.diagonal(cos_similarities)[0]


# Returns a column vector resulting from taking an element-wise mean
def get_avg_vec(vecs):
    return np.mean(vecs, axis=0)


if __name__ == '__main__':

    def basic_test():
        vecs = [np.array([1, 3]).reshape(1, -1), np.array([2, 1]).reshape(1, -1)]
        other_vecs = [np.array([-0.5, 0.5]).reshape(1, -1), np.array([0, 3]).reshape(1, -1)]

        avg_1 = get_avg_vec(vecs)
        avg_2 = get_avg_vec(other_vecs)

        print(avg_1)
        print(avg_2)

        print(cos_sim(vecs, other_vecs))

        claim = 'Hello, my name is frank!'
        article_one = ' '.join([
            'Hello, my name is frank!',
            'hello I am frank. this article is about frank.',
            'semantic similarity is hard to calculate',
            'fruity punch with Pickle and Olives are great!! I am so excited to have punch with oranges and grapes',
            'Mayweather fights his last championship fight tonight. It will be Mayweather versus Canelo',
            'faked news on the rise in the united states greatly. Fake news is a big concern',
            'here are the directions to make the perfect avocado toast'
        ])
        article_two = ' '.join([
            'hello I am frank loool asdfdsalfkxv.',
            'this is a sentence with similarity in semantics and such.',
            'pickles and oranges don\'t go together but pickles and apples do.',
            'stocks are on the rise again.',
            'absolute fake news tooo few characters? 21324s.',
            '22 franks in one room said hello.',
            '===/asdf/af https://google.com source article === asdf/af click here to view more'
        ])
        from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer

        v = GoogleNewsVectorizer(path='./assets/GoogleNews-vectors-negative300.bin.gz')
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 500)
        print(relevant_info)
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 5)
        print(relevant_info)

    def test_from_data():
        from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer
        import pandas as pd
        v = GoogleNewsVectorizer(path='./assets/GoogleNews-vectors-negative300.bin.gz')
        json_data = pd.read_pickle('../data/json_data.pkl')[0:1]  # just test first
        articles_data = pd.read_pickle('../data/articles_data.pkl')
        claims, supp_info, labels = preprocess_nn(json_data, articles_data, vectorizer=v, max_seq_len=500)
        print(claims)
        print(supp_info)
        print(labels)

    test_from_data()