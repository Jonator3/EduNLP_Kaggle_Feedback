import os
from typing import Callable, List

import nltk


def get_file(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return " ".join(lines)


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


def ngram_tokenize(text: str, n: int, preprocess: Callable[[str], str] = None,
                   post_tokenize_process: Callable[[List[str]], List[str]] = None,
                   post_ngram_process: Callable[[List[tuple[str]], int], List[tuple[str]]] = None) -> List[tuple[str]]:
    if preprocess is not None:
        text = preprocess(text)
    tokens = nltk.tokenize.word_tokenize(text)
    if post_tokenize_process is not None:
        tokens = post_tokenize_process(tokens)
    ngrams = []
    for i in range(max(len(tokens) - (n-1), 0)):
        ngram = tuple([tokens[i + j] for j in range(n)])
        ngrams.append(ngram)
    if post_ngram_process is not None:
        ngrams = post_ngram_process(ngrams, n)
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


def load(input_folder: str, clusters=15, n_gram: int = 1, preprocess: Callable[[str], str] = None, post_tokenize_process: Callable[[List[str]], List[str]] = None, post_ngram_process: Callable[[List[tuple[str]]], List[tuple[str]]] = None) -> DataSet:
    if n_gram < 1:
        raise ValueError("n_gram parameter must be int > 0!")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError("input_folder: '" + input_folder + "' does not exist or is not a folder!")
    if not input_folder.endswith("/"):
        input_folder += "/"
    doc_count = 0
    doc_freq = {}
    front_corpora = []
    bcs = []
    for i in range(clusters):
        token = []
        print("loading cluster", i)
        for file in os.listdir(input_folder + str(i)):
            raw_text = get_file(input_folder + str(i) + "/" + file)
            sub_token = ngram_tokenize(raw_text, n_gram, preprocess, post_tokenize_process, post_ngram_process)
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
