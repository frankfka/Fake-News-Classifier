import pandas as pd
import spacy
import re
from nltk import sent_tokenize


def get_trainable_df(json_df, articles_df):
    """
    Expects as input the output of our data loader (json df and articles df)
    Creates a DF that has the columns: claim, support (i.e. supporting text), and label
    """
    spacy_nlp = spacy.load('en_core_web_md')  # Use python -m spacy download en_core_web_md
    # Arrays to create the DF from
    claims = []
    supporting_info = []
    labels = []

    # Loop over each claim and its metadata
    for index, json_data in json_df.iterrows():
        print(f"Processing Claim {index}")
        '''
        Extract basic data from json data
        '''
        # Uncleaned claim sentence
        claim = json_data['claim']
        clean_claim = clean_sent(claim)
        # Integer ID's for the claim
        article_ids = json_data['related_articles']
        # Final supporting text for the claim
        supporting_text = ''
        # Label for the claim
        label = json_data['label']

        '''
        Loop over each article to create the supporting text
        '''
        for article_id in article_ids:
            article_txt = articles_df.loc[str(article_id), 'text']
            article_sents = [clean_sent(sentence) for sentence in sent_tokenize(article_txt)]
            similar_sents = get_most_similar_sentences_spacy(clean_claim, article_sents, 3, spacy_nlp)
            # Just get the sentences (similar_sents includes the similarities)
            supporting_text += " " + " ".join([item[1] for item in similar_sents])

        '''
        Append data to the lists
        '''
        claims.append(clean_claim)
        supporting_info.append(supporting_text)
        labels.append(label)

    '''
    Create the DF with columns: claim, support, label
    '''
    return pd.DataFrame(data={"claim": claims, "support": supporting_info, "label": labels})


def get_most_similar_sentences_spacy(claim, sentences, n, nlp):
    """
    Uses Spacy model to find the n most similar sentences between the claim and the doc
    - Claim: str, sentences: [str], n: int, nlp: spacy model
    - Expects the input strings to be already cleaned & processed
    - Returns array of (similarity, sentence) where sentence is a string
    """
    similarities_and_sents = []
    spacy_claim = nlp(claim)
    for sentence in sentences:
        # Don't process for sentences less than 20 characters long - this usually means improper sentences/words
        # That won't help the learning process
        if len(sentence) < 20:
            continue
        spacy_sent = nlp(sentence)
        similarity = spacy_claim.similarity(spacy_sent)
        similarities_and_sents.append((similarity, sentence))
    # Sort the similarities (in desc order) using their similarity
    sorted_sents = sorted(similarities_and_sents, key=lambda elem: elem[0], reverse=True)
    return sorted_sents[0:n] if n < len(sorted_sents) else sorted_sents


def clean_sent(txt):
    """
    Cleans a string for processing - should be used for sentences since punctuation is removed
    - Replace non alphanumeric with whitespace
    - To lowercase
    - Trim whitespace
    """
    return re.sub(r'\W+', ' ', txt).lower().strip()
