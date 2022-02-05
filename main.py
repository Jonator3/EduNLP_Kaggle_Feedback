import math
from typing import List

import nltk
from nltk.probability import FreqDist
import data_loader as data


def inverse_doc_frequency(term: str, total_num_docs=data.doc_count):
    return total_num_docs / data.doc_freq.get(term)


def tf_idf(term: str, forground: List[str]):
    doc_freq_dist = nltk.FreqDist(forground)
    return doc_freq_dist.freq(term) * math.log2(inverse_doc_frequency(term))

# tf-idf(t, d) := tf(t, d) * idf(t)
# idf(t) := (N/(df(t))
# df(t) := anzahl von documenten mit token t
# tf(t, d) := frequency of t in d = doc_freq_dist.freq(term)


output = open("tf_idf_output.csv", "w")
output.write("term,total_count,tf_idf(0),tf_idf(1),tf_idf(2),tf_idf(3),tf_idf(4),tf_idf(5),tf_idf(6),tf_idf(7),tf_idf(8),tf_idf(9),tf_idf(10),tf_idf(11),tf_idf(12),tf_idf(13),tf_idf(14)\n")

term_count = len(set(data.back_corpus))
for i, term in enumerate(set(data.back_corpus)): #  every word used in at least on of the documents
    print(i, "/", term_count, ":", term)
    count = data.back_corpus.count(term)
    output.write(term+","+str(count))
    for fi in data.front_corpus_dict.keys():
        output.write("," + str(round(tf_idf(term, data.front_corpus_dict.get(fi))*1000000, 1)))
    output.write("\n")
