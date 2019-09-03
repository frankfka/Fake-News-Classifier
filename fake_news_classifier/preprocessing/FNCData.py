import pandas as pd

from fake_news_classifier.const import TEXT_TWO_IDX, TEXT_ONE_IDX, LABEL_IDX, CRED_IDX, CLAIM_ID_IDX
from fake_news_classifier.util import log


# Balances class data to be equal
def balance_classes(df, max_bias=1):
    # Filter by label
    true_claims = df[df[LABEL_IDX] == 2]
    neutral_claims = df[df[LABEL_IDX] == 1]
    false_claims = df[df[LABEL_IDX] == 0]
    # Get the least # of counts
    max_index = min([
        len(true_claims.index),
        len(neutral_claims.index),
        len(false_claims.index),
    ])
    # Get the max # given the bias and the max possible index
    max_index_biased = int(max_bias * max_index)
    # Truncate to the maximum length
    if len(true_claims.index) > max_index_biased:
        true_claims = true_claims[0:max_index_biased]
    if len(neutral_claims.index) > max_index_biased:
        neutral_claims = neutral_claims[0:max_index_biased]
    if len(false_claims.index) > max_index_biased:
        false_claims = false_claims[0:max_index_biased]
    return pd.concat([
        true_claims,
        neutral_claims,
        false_claims
    ]).sample(frac=1)  # This shuffles


class FNCData(object):
    """
    Contains Data from the FNC dataset
    - This class is meant to be fed into a neural network
    - Contains a DF for body1, body2, label (0, 1, or 2)
    - Usually, body1 should be the claim, but this is meant to be universal
    - TODO: Can incorporate credibility in this, if we go that route
    - TODO: Can incorporate preprocessing, once we finalize
    """

    def __init__(self, uids, list_of_txt, other_list_of_txt, list_of_labels, vectorizer, max_seq_len,
                 max_label_bias=None, list_of_cred=None):
        creds = [None] * len(list_of_labels)
        if list_of_cred is not None:
            creds = list_of_cred
        self.data = pd.DataFrame(data={
            CLAIM_ID_IDX: uids,
            TEXT_ONE_IDX: list_of_txt,
            TEXT_TWO_IDX: other_list_of_txt,
            CRED_IDX: creds,
            LABEL_IDX: list_of_labels
        })
        self.vectorizer = vectorizer
        self.max_seq_len = max_seq_len
        log(f"FNCData label counts: \n{self.data[LABEL_IDX].value_counts()}")
        if max_label_bias is not None:
            self.data = balance_classes(self.data, max_label_bias)
            log(f"FNCData labels balanced with max bias {max_label_bias}. " +
                f"New label counts: \n{self.data[LABEL_IDX].value_counts()}")
        log("FNCData Initialized")

    # Get data (text, other_text, labels)
    # limits to certain indicies if provided, transforms using vectorizer if provided
    def get(self, vectorize=False, idx=None, use_ngrams=False):
        sample_data = self.data
        if idx is not None:
            # Limit to certain indicies
            sample_data = sample_data.iloc[idx, :]
        uids = sample_data[CLAIM_ID_IDX]
        texts = sample_data[TEXT_ONE_IDX]
        other_texts = sample_data[TEXT_TWO_IDX]
        creds = sample_data[CRED_IDX]
        sample_labels = sample_data[LABEL_IDX]
        if vectorize:
            return (
                uids,
                self.vectorizer.transform_list_of_txt(texts, self.max_seq_len, use_ngrams=use_ngrams),
                self.vectorizer.transform_list_of_txt(other_texts, self.max_seq_len, use_ngrams=use_ngrams),
                creds,
                sample_labels
            )
        else:
            return (
                uids,
                texts,
                other_texts,
                creds,
                sample_labels
            )


if __name__ == '__main__':
    from fake_news_classifier.preprocessing.Word2VecVectorizer import Word2VecVectorizer
    v = Word2VecVectorizer(path='./assets/GoogleNewsVectors.bin.gz')

    one_list = ["hello hello hellohello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]
    other_list = ["hello hello hello hello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]

    data = FNCData(one_list, other_list, [0, 0], v, 500)
    vec_txt, vec_other_txt, creds, labels = data.get(vectorize=True)

    print(f"Vec Text: {len(vec_txt)} x {len(vec_txt[0])} x {len(vec_txt[0][0])}")
    print(labels)
