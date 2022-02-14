import os
from typing import Callable

import nltk


def get_file_token(file_path: str, tokenize: Callable[[str], list], preprocess: Callable[[str], str]):
    with open(file_path, "r") as file:
        token = []
        for line in file.readlines():
            token += tokenize(preprocess(line))
        return token


class DataLoader:

    def __init__(self,
                 tokenize: Callable[[str], list] = nltk.tokenize.word_tokenize,
                 preprocess: Callable[[str], str] = lambda x: x):
        self.front_corpus_dict = {}
        self.front_corpus_freqdist_dict = {}
        self.back_corpus = []
        self.doc_freq = {}
        self.token_count_dict = {}
        self.doc_count = 0
        self.tokenize = tokenize
        self.preprocess = preprocess

    def load(self, clusters=15):

        for i in range(clusters):
            print("loading cluster", i)
            token = []
            for file in os.listdir("clusters/" + str(i)):
                sub_token = get_file_token("clusters/" + str(i) + "/" + file, self.tokenize, self.preprocess)
                self.doc_count += 1
                token += sub_token
                for t in set(sub_token):
                    if self.doc_freq.get(t) is None:
                        self.doc_freq[t] = 1
                    else:
                        self.doc_freq[t] += 1
            self.front_corpus_dict[i] = token
            self.back_corpus += token
            self.front_corpus_freqdist_dict[i] = nltk.FreqDist(token)
        print("counting tokens")
        bcs = self.back_corpus.copy()
        bcs.sort()
        current_term = None
        segment_start = -1
        for i, term in enumerate(bcs):
            if term != current_term:
                if current_term is not None:
                    self.token_count_dict[current_term] = i - segment_start
                segment_start = i
                current_term = term
        print("loading done")
        print("totaling", self.doc_count, "files")
        print("")
