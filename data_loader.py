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


def load():
    global front_corpus_dict, back_corpus
    front_corpus_dict = {}
    back_corpus = []
    for i in range(15):
        print("loading cluster", i)
        token = []
        for file in os.listdir("clusters/"+str(i)):
            token += get_file_token("clusters/"+str(i)+"/"+file)
        front_corpus_dict[i] = token
        back_corpus += token
    print("loading done")


load()

