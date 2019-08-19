import re
import time

from nltk import ne_chunk, Tree
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import fake_news_classifier.const as const
from fake_news_classifier.preprocessing.text_util import tokenize_by_sent, tokenize_by_word, clean_sentence, \
    analyze_pos, clean_tokenized, keep_alphanumeric
from fake_news_classifier.util import log

"""
This class abstracts away the details of the FNC dataset. This is so that our models can be generalized.
If the dataset changes, only the preprocessors need to change.
"""

# TODO: This method has changed. Need to run again


def preprocess_nn(json_df, articles_df, vectorizer, max_seq_len):
    """
    Given the raw FNC data, return 3 lists of (text, other_text (supporting info), and labels)
        - Claims are appended with claimant
        - Articles are concatenated and the max_seq_len # of most relevant words are appended into supporting info
        - Labels are passed through as is
    """

    # Raw Data
    claims = json_df[const.PKL_CLAIM]
    claimants = json_df[const.PKL_CLAIMANT]
    labels = json_df[const.PKL_LABEL]
    related_articles = json_df[const.PKL_RELATED_ARTICLES]

    # Processed Data
    processed_claims = []
    supporting_info = []

    start_time = time.time()  # Used for tracking only

    '''
    Loop through all the claims and their article ID's
    '''
    for j, (str_claim, str_claimant, article_ids) in enumerate(zip(claims, claimants, related_articles)):

        # Tracking use only
        if j % 1000 == 0 and j != 0:
            now = time.time()
            log(f"Processing claim {j} | Last 1000 claims took {now - start_time} seconds")
            start_time = now

        '''
        Process Claim: 
            Final Claim = Claimant + Claim
            - Take out all non-alphanumeric
            - Keep case - may be important
        '''
        claim = str_claimant + ' ' + str_claim
        claim = keep_alphanumeric(claim)

        '''
        Process articles
            - Get all articles from the dataframe by ID
            - Get relevant info from the article, truncated to max_seq_len # of words
        '''
        # Get list of article bodies from the dataframe
        article_ids = [str(article_id) for article_id in article_ids]  # Need to lookup by string
        # Get the articles with the given article ID's and only extract the text column
        articles = articles_df.loc[articles_df[const.PKL_ARTICLE_ID].isin(article_ids), const.PKL_ARTICLE_TXT]
        support_txt = get_relevant_info(claim, articles, vectorizer, max_seq_len)

        # Add to list
        processed_claims.append(claim)
        supporting_info.append(support_txt)

    return processed_claims, supporting_info, labels


def get_relevant_info(claim, articles, vectorizer, max_seq_len):
    """
    Returns the most relevant sentences relating to a claim using the average vectors of words and cosine similarity
    - Extra whitespace is trimmed and removed
    - Trims all non-alphanumeric
    - Maintains case

    - TODO: Process long sentences by splitting them up
    """

    # Note: expects claim to be already cleaned
    vec_claim = vectorizer.transform_one(claim)  # Claim vector - we'll use this to compare using cosine similarity
    similarities_and_sents = []  # Stores tuples of (cos sim, sentence)

    # Loop through all articles to construct supporting information
    for article in articles:
        sentences = tokenize_by_sent(article)

        '''
        For each sentence, we clean and vectorize, then retrieve the cosine similarity of the claim vs the sentence
        '''
        for sentence in sentences:
            # Only process alphanumeric characters
            sentence = keep_alphanumeric(sentence)
            # Don't process for sentences less than 40 characters long - this usually means improper sentences/words
            if len(sentence) < 40:
                continue
            # Get vector of sentence and find cosine similarity
            vec_sent = vectorizer.transform_one(sentence)
            similarity = cos_sim(vec_claim, vec_sent)
            # See if any named entities/numbers appear together - if so, take the average of similarity & 0.9
            # This raises the similarity
            if has_ner_or_number_naive(claim, sentence):
                similarity = np.mean([0.9, similarity])
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


# Check if sent_2 contains any named entities or numbers in sent_1
# Naive because it does no NER
def has_ner_or_number(sent_1, sent_2):
    # Get number and NER's from sent_1
    alphanumeric_sent_1 = keep_alphanumeric(sent_1)
    sent_1_toks = tokenize_by_word(alphanumeric_sent_1)
    sent_1_toks = clean_tokenized(sent_1_toks, remove_punctuation=True, remove_stopwords=True)
    sent_1_toks = analyze_pos(sent_1_toks, lemmatize=False)
    # Get numbers
    numbers = list(filter(lambda tok_with_pos: tok_with_pos[1] == 'CD', sent_1_toks))  # has pos tags as tuple
    numbers = [number[0] for number in numbers]  # no pos tags, just numbers

    print(numbers)

    # Get named entities
    def get_named_entities(tokens):
        chunked = ne_chunk(sent_1_toks)
        continuous_chunk = []
        current_chunk = []
        for chunk in chunked:
            if type(chunk) == Tree:
                current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk
    named_entities = list(get_named_entities(sent_1_toks))
    print(named_entities)
    named_entities = [entity.split() for entity in named_entities]
    toks_of_interest = numbers + named_entities
    print(toks_of_interest)
    lowercase_sent_2 = sent_2.lower()
    for tok in toks_of_interest:
        # TODO: this matches a seq of characters (not words) so 'is' will match 'his'
        if tok in lowercase_sent_2:
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

        v = GoogleNewsVectorizer(path='./assets/GoogleNewsVectors.bin.gz')
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 500)
        print(relevant_info)
        relevant_info = get_relevant_info(claim, [article_one, article_two], v, 5)
        print(relevant_info)


    def test_from_data():
        from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer
        import pandas as pd
        v = GoogleNewsVectorizer(path='./assets/GoogleNewsVectors.bin.gz')
        json_data = pd.read_pickle('../data/json_data.pkl')[0:1]  # just test first
        articles_data = pd.read_pickle('../data/articles_data.pkl')
        claims, supp_info, labels = preprocess_nn(json_data, articles_data, vectorizer=v, max_seq_len=500)
        print(claims)
        print(supp_info)
        print(labels)


    test_from_data()
