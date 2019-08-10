import gensim
import time
import numpy as np
from fake_news_classifier.preprocessing.text_util import tokenize_by_word
from fake_news_classifier.util.misc import log

DEFAULT_MAX_SEQ_LEN = 500


class GoogleNewsVectorizer(object):
    """
    Google Vectorization object that allows for text -> vector. Uses pre-trained GoogleNews vectors
    """

    def __init__(self, path='./preprocessing/assets/GoogleNews-vectors-negative300.bin.gz'):
        start_time = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, unicode_errors='ignore', binary=True)
        log(f"Google word vectors loaded in {time.time() - start_time}s")

    def transform_many(self, list_of_txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        return [
            self.transform_one(txt, max_seq_len=max_seq_len) for txt in list_of_txt
        ]

    def transform_one(self, txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        # Tokenize text into words, then into vectors
        words = tokenize_by_word(txt)
        words = words[0:max_seq_len] if len(words) > max_seq_len else words
        return [self.get_word_vec(word) for word in words]

    def get_word_vec(self, word):
        # Best possible case - word in model, return that vector
        if word in self.model.vocab:
            return self.model[word]
        # Try a lowercase representation
        if word.lower() in self.model.vocab:
            return self.model[word.lower()]
        # Just return an empty vector
        return np.zeros(300, dtype='float32')


if __name__ == '__main__':
    gv = GoogleNewsVectorizer(path='./assets/GoogleNews-vectors-negative300.bin.gz')

    sent = 'the quick brown fox jumped over the lazy dog'
    t = [sent, 'the. quick, brown! fox,, !']
    transformed = gv.transform_many(t)
    transformed_one = gv.transform_one(sent)
    print(transformed)
    assert(transformed[0] == transformed_one)
    print(len(transformed))
    print(len(transformed[0]))
    print(len(transformed[0][0]))
