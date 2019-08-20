import gensim
import time
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from fake_news_classifier.preprocessing.text_util import tokenize_by_word
from fake_news_classifier.util import log

DEFAULT_MAX_SEQ_LEN = 500
WORD2VEC_TMP_FILE = './assets/300d.commoncrawl.converted.glove.txt'


# Helper function to convert GloVe -> word2vec format
def glove_to_word2vec(glove_path, convert_path):
    _ = glove2word2vec(glove_path, convert_path)
    return convert_path


class GloveVectorizer(object):
    """
    Glove Vectorization object
    """

    def __init__(self, word2vec_path=None, glove_path=None):
        start_time = time.time()
        if word2vec_path is not None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
        elif glove_path is not None and word2vec_path is not None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(glove_to_word2vec(glove_path, word2vec_path))
        else:
            raise ValueError("Required params not passed")
        log(f"GloVe word vectors loaded in {time.time() - start_time}s")

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
        # Separator - return ones
        if word == "|SEP|":
            return np.ones(300, dtype='float32')
        # Best possible case - word in model, return that vector
        if word in self.model.vocab:
            return self.model[word]
        # Try a lowercase representation
        if word.lower() in self.model.vocab:
            return self.model[word.lower()]
        # Just return an empty vector
        return np.zeros(300, dtype='float32')


if __name__ == '__main__':
    gv = GloveVectorizer(word2vec_path='./assets/300d.commoncrawl.converted.glove.txt')

    sent = 'the quick brown fox jumped over the lazy dog'
    t = [sent, 'the. quick, brown! fox,, !']
    transformed = gv.transform_many(t)
    transformed_one = gv.transform_one(sent)
    print(transformed)
    assert(transformed[0] == transformed_one)
    print(len(transformed))
    print(len(transformed[0]))
    print(len(transformed[0][0]))
