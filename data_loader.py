import os
from typing import Callable, List

import nltk


def get_file_token(file_path: str, n_gram, preprocess: Callable[[str], str]):
    with open(file_path, "r") as file:
        token = []
        for line in file.readlines():
            token += ngram_tokenize(preprocess(line), n_gram)
        return token


def count_token(corpus):
    corpus = corpus.copy()
    corpus.sort()
    token_count = {}
    current_term = None
    segment_start = -1
    for i, term in enumerate(corpus):
        if term != current_term:
            if current_term is not None:
                token_count[current_term] = i - segment_start
            segment_start = i
            current_term = term
    return token_count


def ngram_tokenize(text: str, n: int) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    ngrams = []
    for i in range(max(len(tokens) - (n-1), 0)):
        ngram = tuple([tokens[i + j] for j in range(n)])
        ngrams.append(ngram)
    return ngrams


class DataSet:

    def __init__(self, fc, df, dc, tc, n):
        self.n_gram = n
        self.front_corpora = fc
        self.doc_freq = df
        self.doc_count = dc
        self.token_count = tc
        self.front_corpora_freqdist = []
        for F in self.front_corpora:
            self.front_corpora_freqdist.append(nltk.FreqDist(F))

    def get_back_corpus(self):
        bkc = []
        for F in self.front_corpora:
            bkc += F
        return bkc


def load(clusters=15, n_gram: int = 1, preprocess: Callable[[str], str] = lambda x: x) -> DataSet:
    if n_gram < 1:
        raise ValueError("n_gram parameter must be int > 0!")
    doc_count = 0
    doc_freq = {}
    front_corpora = []
    bcs = []
    for i in range(clusters):
        token = []
        print("loading cluster", i)
        for file in os.listdir("clusters/" + str(i)):
            sub_token = get_file_token("clusters/" + str(i) + "/" + file, n_gram, preprocess)
            doc_count += 1
            token += sub_token
            for t in set(sub_token):
                if doc_freq.get(t) is None:
                    doc_freq[t] = 1
                else:
                    doc_freq[t] += 1
        front_corpora.append(token)
        bcs += token
    print("counting tokens")
    token_count = count_token(bcs)
    bcs = None
    del bcs
    print("making Frequency distributions")
    data = DataSet(front_corpora, doc_freq, doc_count, token_count, n_gram)
    print("loading done")
    print("totaling", doc_count, "files")
    print("")
    return data
