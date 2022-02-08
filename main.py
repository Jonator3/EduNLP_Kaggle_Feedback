import datetime
import math
from typing import List

import nltk
from nltk.probability import FreqDist
import data_loader as data


def inverse_doc_frequency(term: str, total_num_docs=None):
    if total_num_docs is None:
        total_num_docs = data.doc_count
    return total_num_docs / data.doc_freq.get(term) + 1


def tf_idf(term: str, forground_freqdist: nltk.FreqDist):
    return forground_freqdist.freq(term) * math.log2(inverse_doc_frequency(term, 15))

# tf-idf(t, d) := tf(t, d) * idf(t)
# idf(t) := log2(N/(df(t)+1)
# df(t) := anzahl von documenten mit token t
# tf(t, d) := frequency of t in d = doc_freq_dist.freq(term)


def calc_all_tfidf(min_count=5):
    output = open("tf_idf_output.csv", "w")
    output.write("term,total_count,tf_idf(0),tf_idf(1),tf_idf(2),tf_idf(3),tf_idf(4),tf_idf(5),tf_idf(6),tf_idf(7),tf_idf(8),tf_idf(9),tf_idf(10),tf_idf(11),tf_idf(12),tf_idf(13),tf_idf(14)\n")

    tokens = list(set(data.back_corpus))
    tokens.sort()
    term_count = len(tokens)
    for i, term in enumerate(tokens):  # every word used in at least on of the documents
        count = data.token_count_dict.get(term)
        if count is None:
            continue
        if count < min_count:
            continue
        print(i, "/", term_count, ":", term)
        output.write('"' + term + '",' + str(count))
        for fi in data.front_corpus_freqdist_dict.keys():
            tf_idf_val = tf_idf(term, data.front_corpus_freqdist_dict.get(fi))
            output.write("," + format(tf_idf_val, ".15f"))
        output.write("\n")


def calc_count_dist():
    count_dist = {}
    for term in set(data.back_corpus):
        count = data.token_count_dict.get(term)
        if count_dist.get(count) is None:
            count_dist[count] = 1
        else:
            count_dist[count] += 1
    output = open("token_count_distribution_output.txt", "w")

    for i in range(1000):
        count = count_dist.get(i+1)
        if count is None:
            count = 0
        print(i+1, ":", count)
        output.write(str(count) + "\n")


start_time = datetime.datetime.now()
data.load()

calc_all_tfidf()

print("\nTime:", datetime.datetime.now() - start_time)
