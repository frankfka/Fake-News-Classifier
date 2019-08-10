import pandas as pd

from fake_news_classifier.util.misc import log

TEXT_ONE_IDX = 'text_1'
TEXT_TWO_IDX = 'text_2'
LABEL_IDX = 'label'


class FNCData(object):
    """
    Contains Data from the FNC dataset
    - This class is meant to be fed into a neural network
    - Contains a DF for body1, body2, label (0, 1, or 2)
    - Usually, body1 should be the claim, but this is meant to be universal
    """

    def __init__(self, list_of_txt, other_list_of_txt, list_of_labels, vectorizer, max_seq_len):
        self.data = pd.DataFrame(data={
            TEXT_ONE_IDX: list_of_txt,
            TEXT_TWO_IDX: other_list_of_txt,
            LABEL_IDX: list_of_labels
        })
        self.vectorizer = vectorizer
        self.max_seq_len = max_seq_len
        log(f"FNCData label counts: {self.data[LABEL_IDX].value_counts()}")
        log("FNCData Initialized")

    # Get data (text, other_text, labels)
    # limits to certain indicies if provided, transforms using vectorizer if provided
    def get(self, vectorize=False, idx=None):
        data = self.data
        if idx is not None:
            # Limit to certain indicies
            data = data.loc[idx, :]
        texts = data[TEXT_ONE_IDX]
        other_texts = data[TEXT_TWO_IDX]
        labels = data[LABEL_IDX]
        if vectorize:
            return (
                self.vectorizer.transform_many(texts, self.max_seq_len),
                self.vectorizer.transform_many(other_texts, self.max_seq_len),
                labels
            )
        else:
            return (
                texts,
                other_texts,
                labels
            )


if __name__ == '__main__':
    pass
    # TODO: Test this method
