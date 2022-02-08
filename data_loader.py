import os

import nltk


def get_file_token(file_path):
    with open(file_path, "r") as file:
        token = []
        for line in file.readlines():
            token += nltk.tokenize.word_tokenize(line)
        return token


front_corpus_dict = {}
back_corpus = []
doc_freq = {}
doc_count = 0
token_count_dict = {}
front_corpus_freqdist_dict = {}

def load(clusters=15):
    global front_corpus_dict, back_corpus, doc_count, doc_freq, token_count_dict, front_corpus_freqdist_dict

    # reset all values
    front_corpus_dict = {}
    front_corpus_freqdist_dict = {}
    back_corpus = []
    doc_freq = {}
    token_count_dict = {}
    doc_count = 0

    for i in range(clusters):
        print("loading cluster", i)
        token = []
        for file in os.listdir("clusters/"+str(i)):
            sub_token = get_file_token("clusters/"+str(i)+"/"+file)
            doc_count += 1
            token += sub_token
            for t in set(sub_token):
                if doc_freq.get(t) is None:
                    doc_freq[t] = 1
                else:
                    doc_freq[t] += 1
        front_corpus_dict[i] = token
        back_corpus += token
        front_corpus_freqdist_dict[i] = nltk.FreqDist(token)
    print("counting tokens")
    bcs = back_corpus.copy()
    bcs.sort()
    current_term = None
    segment_start = -1
    for i, term in enumerate(bcs):
        if term != current_term:
            if current_term is not None:
                token_count_dict[current_term] = i - segment_start
            segment_start = i
            current_term = term
    print("loading done")

