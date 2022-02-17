import datetime
import math
import sys

import nltk

from preprocess import *
import tokenizers
import data_loader





def inverse_doc_frequency(term: str, data):
    total_num_docs = data.doc_count
    return total_num_docs / (data.doc_freq.get(term) + 1)


def tf_idf(term: str, fg_fd: nltk.FreqDist, data):
    return fg_fd.freq(term) * math.log(inverse_doc_frequency(term, data))


# tf-idf(t, d) := tf(t, d) * idf(t)
# idf(t) := log2(N/(df(t)+1))
# df(t) := anzahl von documenten mit token t
# tf(t, d) := frequency of t in d = doc_freq_dist.freq(term)


def calc_all_tfidf(data, min_count=5):
    output = open("tf_idf_"+str(data.n_gram)+"_output.csv", "w")
    output.write(
        "term,total_count,tf_idf(0),tf_idf(1),tf_idf(2),tf_idf(3),tf_idf(4),tf_idf(5),tf_idf(6),tf_idf(7),tf_idf(8),tf_idf(9),tf_idf(10),tf_idf(11),tf_idf(12),tf_idf(13),tf_idf(14)\n")

    tokens = list(set(data.get_back_corpus()))
    tokens.sort()
    print("calculating tf_idf values")
    for i, term in enumerate(tokens):  # every word used in at least on of the documents
        count = data.token_count.get(term)
        if count is None:
            continue
        if count < min_count:
            continue
        output.write('"' + str(term) + '",' + str(count))
        for fcf in data.front_corpora_freqdist:
            tf_idf_val = tf_idf(term, fcf, data)
            output.write("," + format(tf_idf_val, ".15f"))
        output.write("\n")
    print("tf_idf done")


def calc_count_dist(data):
    count_dist = {}
    for term in set(data.get_back_corpus()):
        count = data.token_count.get(term)
        if count_dist.get(count) is None:
            count_dist[count] = 1
        else:
            count_dist[count] += 1
    output = open("token_count_distribution_"+str(data.n_gram)+"_output.txt", "w")

    for i in range(1000):
        count = count_dist.get(i + 1)
        if count is None:
            count = 0
        print(i + 1, ":", count)
        output.write(str(count) + "\n")


for n in range(1, 4):
    start_time = datetime.datetime.now()
    print("running", str(n) + "-gram", start_time)

    data = data_loader.load(n_gram=n, preprocess=compose(lower, remove_quotes, remove_punctuation))

    calc_all_tfidf(data)

    print("\nTime:", datetime.datetime.now() - start_time)
