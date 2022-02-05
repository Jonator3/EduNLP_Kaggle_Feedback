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


def load():
    global front_corpus_dict, back_corpus, doc_count, doc_freq
    front_corpus_dict = {}
    back_corpus = []
    doc_freq = {}
    doc_count = 0
    for i in range(15):
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
    print("loading done")


load()

