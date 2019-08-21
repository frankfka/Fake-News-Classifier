import gensim
import time
import numpy as np
from nltk import ngrams

from fake_news_classifier.preprocessing.text_util import tokenize_by_word
from fake_news_classifier.util import log

DEFAULT_MAX_SEQ_LEN = 500


class Word2VecNGramsVectorizer(object):
    """
    Google Vectorization object that allows for text -> vector. Uses pre-trained GoogleNews vectors
    """

    def __init__(self, path='./preprocessing/assets/GoogleNewsVectors.bin.gz'):
        start_time = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, unicode_errors='ignore', binary=True)
        log(f"Word2Vec vectors loaded in {time.time() - start_time}s")

    def transform_many(self, list_of_txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        return [
            self.transform_one(txt, max_seq_len=max_seq_len) for txt in list_of_txt
        ]

    def transform_one(self, txt, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        # Tokenize text into words, then into vectors
        words = tokenize_by_word(txt)
        # Note: the final list may be shorter than max seq len if bi/tri-grams are found
        words = words[0:max_seq_len] if len(words) > max_seq_len else words
        num_words = len(words)
        vectors = []  # Stores word vectors
        idx = 0  # We'll loop over the entire words list
        while idx < num_words:
            # Try tri-grams (in_this_format) if index allows
            if idx + 2 < num_words:
                trigram = words[idx] + '_' + words[idx + 1] + '_' + words[idx + 2]
                vec = self.get_word_vec(trigram)
                # If vector is found, append to list, update index, and skip to next iteration of loop
                if vec is not None:
                    vectors.append(vec)
                    idx += 3
                    continue  # Don't consider bi-grams
            # Try bi-grams
            if idx + 1 < num_words:
                bigram = words[idx] + '_' + words[idx + 1]
                vec = self.get_word_vec(bigram)
                if vec is not None:
                    vectors.append(vec)
                    idx += 2
                    continue
            # Default to uni-gram
            vec = self.get_word_vec(words[idx])
            if vec is not None:
                vectors.append(vec)
            else:
                # Default to a zero vector
                vectors.append(np.zeros(300, dtype='float32'))
            idx += 1

        return vectors

    # Word can be a n-gram as well
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
        # Just return None
        return None


if __name__ == '__main__':
    gv = Word2VecNGramsVectorizer(path='./assets/GoogleNewsVectors.bin.gz')

    sent = 'the quick brown fox jumped over the lazy dog'
    t = [sent, 'the. quick, brown! fox,, !']
    transformed = gv.transform_many(t)
    transformed_one = gv.transform_one(sent)
    print(transformed)
    assert(transformed[0] == transformed_one)
    print(len(transformed))
    print(len(transformed[0]))
    print(len(transformed[0][0]))
