import re
import time

from symspellpy.symspellpy import SymSpell

from nltk import ne_chunk, Tree
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import fake_news_classifier.const as const
from fake_news_classifier.preprocessing.text_util import tokenize_by_sentence, tokenize_by_word, clean_sentence, \
    analyze_pos, clean_tokenized, keep_alphanumeric, convert_nums_to_words
from fake_news_classifier.util import log

"""
This class abstracts away the details of the FNC dataset. This is so that our models can be generalized.
If the dataset changes, only the preprocessors need to change.
"""


def preprocess_nn(json_df, articles_df, vectorizer, max_seq_len, use_ngrams=True, credibility_model=None):
    """
    Given the raw FNC data, return 3 lists of (text, other_text (supporting info), and labels)
        - Claims are appended with claimant
        - Articles are concatenated and the max_seq_len # of most relevant words are appended into supporting info
        - Labels are passed through as is

    if credibility_model is not None, will return an additional 'credibilities' list of 1's and 0's
        - Evaluated from non-processed data
    """

    # Raw Data
    claims = json_df[const.PKL_CLAIM]
    claimants = json_df[const.PKL_CLAIMANT]
    labels = json_df[const.PKL_LABEL]
    related_articles = json_df[const.PKL_RELATED_ARTICLES]

    # Processed Data
    processed_claims = []
    supporting_info = []
    credibilities = []
    final_labels = []

    start_time = time.time()  # Used for tracking only

    '''
    Loop through all the claims and their article ID's
    '''
    for j, (str_claim, str_claimant, article_ids, label) in enumerate(zip(claims, claimants, related_articles, labels)):

        # Tracking use only
        if j % 1000 == 0 and j != 0:
            now = time.time()
            log(f"Processing claim {j} | Last 1000 claims took {now - start_time} seconds")
            start_time = now

        '''
        Process Claim: 
            Final Claim = Claimant + Claim
            - Convert numbers to string representation
            - Take out all non-alphanumeric
            - Keep case - may be important
        '''
        claim = str_claimant + ' ' + str_claim
        claim = clean_txt(claim)

        log(f"Claim: {claim}")

        '''
        Process articles
            - Get all articles from the dataframe by ID
            - Get relevant info from the article, truncated to max_seq_len # of words
        '''
        # Get list of article bodies from the dataframe
        article_ids = [str(article_id) for article_id in article_ids]  # Need to lookup by string
        # Get the articles with the given article ID's and only extract the text column
        articles = articles_df.loc[articles_df[const.PKL_ARTICLE_ID].isin(article_ids), const.PKL_ARTICLE_TXT]

        # If using credibility, we separate the articles
        if credibility_model is not None:
            for article in articles:
                credibility = get_credibility(article, credibility_model)
                support_txt = get_relevant_info(claim, [article], vectorizer, max_seq_len, use_ngrams)
                # Add to list
                credibilities.append(credibility)
                processed_claims.append(claim)
                supporting_info.append(support_txt)
                final_labels.append(label)

        else:
            # If we are not using credibility model, construct support text from all articles
            support_txt = get_relevant_info(claim, articles, vectorizer, max_seq_len, use_ngrams)
            # Add to list
            processed_claims.append(claim)
            supporting_info.append(support_txt)
            final_labels.append(label)

    # Return what's appropriate
    if credibility_model is not None:
        return processed_claims, supporting_info, final_labels, credibilities
    else:
        return processed_claims, supporting_info, final_labels


def get_relevant_info(claim, articles, vectorizer, max_seq_len, use_ngrams):
    """
    Returns the most relevant sentences relating to a claim using the average vectors of words and cosine similarity
    - Extra whitespace is trimmed and removed
    - Trims all non-alphanumeric
    - Maintains case

    - TODO: Process long sentences by splitting them up
    """

    # Note: expects claim to be already cleaned
    vec_claim = vectorizer.transform_txt(claim, max_seq_len,
                                         use_ngrams=use_ngrams)  # Claim vector - we'll use this to compare using cosine similarity
    similarities_and_sents = []  # Stores tuples of (cos sim, sentence)

    # Loop through all articles to construct supporting information
    for article in articles:
        sentences = tokenize_by_sentence(article)

        '''
        For each sentence, we clean and vectorize, then retrieve the cosine similarity of the claim vs the sentence
        '''
        for sentence in sentences:
            # Basic cleaning on sentence
            sentence = clean_txt(sentence)
            # Don't process for sentences less than 40 characters long - this usually means improper sentences/words
            if len(sentence) < 40:
                continue
            # Get vector of sentence and find cosine similarity
            vec_sent = vectorizer.transform_txt(sentence, max_seq_len,
                                                use_ngrams=use_ngrams)
            similarity = cos_sim(vec_claim, vec_sent)
            # Add to results
            similarities_and_sents.append((similarity, sentence))

    # Sort the similarities (in desc order) using their similarity
    sorted_sents = sorted(similarities_and_sents, key=lambda elem: elem[0], reverse=True)

    article_info = ''
    num_words = 0
    '''
    Construct relevant info - keep looping through sentences, adding sentences until we hit max_seq_len
    We'll surpass max_seq_len, but that's okay
    '''
    for similarity, sentence in sorted_sents:
        if num_words >= max_seq_len:
            break
        article_info += ' |SEP| ' + sentence  # Add a separator
        num_words += len(sentence.split())
    return article_info


# Check if sent_2 contains any named entities or numbers in sent_1
# Naive because it does no NER
# TODO: not being used - not sure if its too useful
def has_ner_or_number_naive(sent_1, sent_2):
    # Just keep alphanumerics
    alphanumeric_sent_1 = keep_alphanumeric(sent_1)
    # Tokenize, remove stopwords, and remove punctuation
    sent_1_toks = tokenize_by_word(alphanumeric_sent_1)
    sent_1_toks = clean_tokenized(sent_1_toks, remove_punctuation=True, remove_stopwords=True)
    # Find POS
    sent_1_toks = analyze_pos(sent_1_toks, lemmatize=False)
    # CD -> number, NNP/NNPS -> proper nouns
    pos_of_interest = {'CD', 'NNP', 'NNPS'}
    toks_of_interest = list(filter(lambda tok_with_pos: tok_with_pos[1] in pos_of_interest, sent_1_toks))
    toks_of_interest = [tok_with_pos[0].lower() for tok_with_pos in toks_of_interest]

    # Tokenize sentence 2 by word, then case to lower case
    sent_2 = keep_alphanumeric(sent_2)
    sent_2_toks = [word.lower() for word in tokenize_by_word(sent_2)]

    # If any tokens of interest are in the second sentence, return true
    for tok in toks_of_interest:
        if tok in sent_2_toks:
            return True
    return False


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


# Returns 0/1 credibility from ArticleCredibilityPAC
def get_credibility(article, credibility_model):
    return credibility_model.predict([article], predict_args={})[0]


def clean_txt_new_methods(txt):
    # Spell check last
    checker = get_spellchecker()
    txt = correct_spelling(txt, checker)

    return txt


def get_spellchecker(dict_path='./assets/spell_check_dictionary.txt'):
    checker = SymSpell()
    checker.load_dictionary(dict_path, 0, 1)
    return checker


def correct_spelling(txt, checker, max_edit_dist=2):
    suggestions = checker.lookup_compound(
        phrase=txt,
        max_edit_distance=max_edit_dist,
        ignore_non_words=True,
        transfer_casing=True
    )
    return ' '.join([suggestion.term for suggestion in suggestions])


# Does basic preprocessing on a string sentence to make it vectorization friendly
# - Converts numbers -> word representation (42 to fourty two)
# - Strips all except for alphanumeric
def clean_txt(txt):
    txt = convert_nums_to_words(txt)
    txt = keep_alphanumeric(txt)
    return txt


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
        from fake_news_classifier.preprocessing.Word2VecVectorizer import Word2VecVectorizer

        v = Word2VecVectorizer(path='./assets/GoogleNewsVectors.bin.gz')
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 500)
        print(relevant_info)
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 5)
        print(relevant_info)


    def test_from_data():
        from fake_news_classifier.preprocessing.Word2VecVectorizer import Word2VecVectorizer
        import pandas as pd
        v = Word2VecVectorizer(path='./assets/GoogleNewsVectors.bin.gz')
        json_data = pd.read_pickle('../data/json_data.pkl')[0:1]  # just test first
        articles_data = pd.read_pickle('../data/articles_data.pkl')
        claims, supp_info, labels = preprocess_nn(json_data, articles_data, vectorizer=v, max_seq_len=500)
        print(claims)
        print(supp_info)
        print(labels)


    test_from_data()
