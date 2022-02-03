import math
from typing import List

import nltk
from nltk.probability import FreqDist
import data_loader as data




# TODO: might be false
def term_frequency(doc_freq_dist: FreqDist, term: str):
    return doc_freq_dist.freq(term)# / doc_freq_dist.freq(doc_freq_dist.max())


def inverse_doc_frequency(doc_freq, total_num_docs=15):
    return total_num_docs / doc_freq


def tf_idf(term: str, forground: List[str], background: List[str]):
    doc_freq_dist = nltk.FreqDist(forground)
    return doc_freq_dist.freq(term) * math.log2(inverse_doc_frequency(background.count(term)+1))

# tf-idf(t, d) := tf(t, d) * idf(t)
# idf(t) := (N/(df(t))
# df(t) := anzahl von t inm document
# tf(t, d) := frequency of t in d = doc_freq_dist.freq(term)

print(tf_idf("Venus", data.front_corpus_dict.get(0), data.back_corpus))
print(tf_idf("Mars", data.front_corpus_dict.get(0), data.back_corpus))
print(tf_idf("the", data.front_corpus_dict.get(0), data.back_corpus))

print(tf_idf("Venus", data.front_corpus_dict.get(1), data.back_corpus))
print(tf_idf("Mars", data.front_corpus_dict.get(1), data.back_corpus))
print(tf_idf("the", data.front_corpus_dict.get(1), data.back_corpus))
