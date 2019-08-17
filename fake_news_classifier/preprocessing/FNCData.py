import pandas as pd

from fake_news_classifier.const import TEXT_TWO_IDX, TEXT_ONE_IDX, LABEL_IDX
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
    """

    def __init__(self, list_of_txt, other_list_of_txt, list_of_labels, vectorizer, max_seq_len, max_label_bias=None):
        self.data = pd.DataFrame(data={
            TEXT_ONE_IDX: list_of_txt,
            TEXT_TWO_IDX: other_list_of_txt,
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
    def get(self, vectorize=False, idx=None):
        sample_data = self.data
        if idx is not None:
            # Limit to certain indicies
            sample_data = sample_data.iloc[idx, :]
        texts = sample_data[TEXT_ONE_IDX]
        other_texts = sample_data[TEXT_TWO_IDX]
        sample_labels = sample_data[LABEL_IDX]
        if vectorize:
            return (
                self.vectorizer.transform_many(texts, self.max_seq_len),
                self.vectorizer.transform_many(other_texts, self.max_seq_len),
                sample_labels
            )
        else:
            return (
                texts,
                other_texts,
                sample_labels
            )


if __name__ == '__main__':
    from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer
    v = GoogleNewsVectorizer(path='./assets/GoogleNewsVectors.bin.gz')

    one_list = ["hello hello hellohello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]
    other_list = ["hello hello hello hello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]

    data = FNCData(one_list, other_list, [0, 0], v, 500)
    vec_txt, vec_other_txt, labels = data.get(vectorize=True)

    print(f"Vec Text: {len(vec_txt)} x {len(vec_txt[0])} x {len(vec_txt[0][0])}")
    print(labels)
