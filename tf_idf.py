import datetime
import math
from argparse import ArgumentParser

import nltk

from preprocessing import *
import data_loader

closed_class_ptags = ["Cd", "CC", "DT", "EX", "IN", "LS", "MD", "PDT", "POS",
                      "PRP", "PRP$", "RP", "TO", "UH", "WDT", "WP", "WP$", "WRB"]


def inverse_doc_frequency(term: str, data):
    total_num_docs = data.doc_count
    return total_num_docs / (data.doc_freq.get(term) + 1)


def tf_idf(term: str, fg_fd: nltk.FreqDist, data):
    return fg_fd.freq(term) * math.log(inverse_doc_frequency(term, data))


def filter_closed_class_words(words: list[str]):
    tagged_words = nltk.pos_tag(words)
    return [tagged_word[0] for tagged_word in tagged_words if tagged_word[1] not in closed_class_ptags]


# tf-idf(t, d) := tf(t, d) * idf(t)
# idf(t) := log2(N/(df(t)+1))
# df(t) := anzahl von documenten mit token t
# tf(t, d) := frequency of t in d = doc_freq_dist.freq(term)


def calc_all_tfidf(data, min_count=5, output_path=None):
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


if __name__ == '__main__':
    arg_pars = ArgumentParser()
    arg_pars.add_argument("input_folder", help="The path to the directory containing the clusters", default=None)
    arg_pars.add_argument("n", help="n-gram length", type=int, default=None)
    args = arg_pars.parse_args()

    input_folder = args.input_folder  # path to the input clusters
    if input_folder is None:
        input_folder = input_folder("Enter the path to the input clusters:\n")
        print("")

    n = args.n  # n for n-grams
    if n is None:
        n = input("Enter n-gram length:\n")
        print("")
        while not n.isdigit():
            n = input("Input must be an Integer!\nEnter n-gram length:\n")
            print("")
        n = int(n)

    start_time = datetime.datetime.now()
    print("\nrunning", str(n) + "-gram", start_time)

    data = data_loader.load(
        input_folder=input_folder,
        n_gram=n,
        preprocess=compose(lower, remove_quotes, remove_punctuation),
        post_tokenize_process=filter_closed_class_words
    )

    calc_all_tfidf(data)

    print("\nTime:", datetime.datetime.now() - start_time)
