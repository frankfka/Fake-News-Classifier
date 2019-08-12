import pandas as pd

from fake_news_classifier.const import TEXT_TWO_IDX, TEXT_ONE_IDX, LABEL_IDX
from fake_news_classifier.util import log


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
        log(f"FNCData label counts: \n{self.data[LABEL_IDX].value_counts()}")
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
    from fake_news_classifier.preprocessing.GoogleNewsVectorizer import GoogleNewsVectorizer
    v = GoogleNewsVectorizer(path='./assets/GoogleNews-vectors-negative300.bin.gz')

    one_list = ["hello hello hellohello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]
    other_list = ["hello hello hello hello hello hello hello", "bye bye bye bye bye bye bye bye bye bye bye "]

    data = FNCData(one_list, other_list, [0, 0], v, 500)
    vec_txt, vec_other_txt, labels = data.get(vectorize=True)

    print(f"Vec Text: {len(vec_txt)} x {len(vec_txt[0])} x {len(vec_txt[0][0])}")
    print(labels)
