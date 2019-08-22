import gensim
import time
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec

from fake_news_classifier.preprocessing.text_util import tokenize_by_word
from fake_news_classifier.util import log


# Helper function to convert GloVe -> word2vec format
def glove_to_word2vec(glove_path, convert_path):
    _ = glove2word2vec(glove_path, convert_path)
    return convert_path


class GensimVectorizer(object):
    """
    Loads a word2vec format from a given path
    - binary: whether to load a binary format (ex. GoogleNewsVectors.bin.gz)
    """

    def __init__(self, path, binary):
        start_time = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path, unicode_errors='ignore', binary=binary)
        log(f"Gensim vectors loaded in {time.time() - start_time}s")

    def transform_list_of_txt(self, list_of_txt, max_seq_len, use_ngrams):
        return [
            self.transform_txt(txt, max_seq_len=max_seq_len, use_ngrams=use_ngrams) for txt in list_of_txt
        ]

    def transform_txt(self, txt, max_seq_len, use_ngrams):
        """
        - Split string into words
        - Iterate over words, if we use n_grams, construct n-grams and see if vectors exist for them
        - For each word, attempt to get vector from the model
        - Separators -> will give an array of ones [1, 1, 1, 1]
        - Not-In-Vocab -> array of zeros
        """

        words = tokenize_by_word(txt)
        words = words[0:max_seq_len] if len(words) > max_seq_len else words

        num_words = len(words)
        vectors = []
        idx = 0

        while idx < num_words:
            # Try tri-grams (in_this_format) if index allows
            if idx + 2 < num_words and use_ngrams:
                trigram = words[idx] + '_' + words[idx + 1] + '_' + words[idx + 2]  # ex. new_york_city
                vec = self.get_word_vec(trigram)
                # If vector is found, append to list, update index, and skip to next iteration of loop
                if vec is not None:
                    vectors.append(vec)
                    idx += 3
                    continue  # Don't consider bi-grams/uni-grams
            # Try bi-grams
            if idx + 1 < num_words:
                bigram = words[idx] + '_' + words[idx + 1]  # ex. donald_trump
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
            print(f"{word} found")
            return self.model[word]
        # Try a lowercase representation
        if word.lower() in self.model.vocab:
            return self.model[word.lower()]
        # Return none if not found
        return None


if __name__ == '__main__':
    gv = GensimVectorizer(path='./assets/GoogleNewsVectors.bin.gz')

    sent = 'the quick brown fox jumped over the lazy dog'
    transformed_one = gv.transform_txt(sent)
    print(len(transformed_one))  # number of vectors
    bigram_sent = 'donald trump!'
    transformed_one = gv.transform_txt(bigram_sent)
    print(len(transformed_one))  # number of vectors
