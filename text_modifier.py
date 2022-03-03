import csv
import datetime
import sys
from typing import Callable, List
from operator import xor
import subprocess

import nltk
import os

import data_loader
import preprocessing


def parse_ngram(text: str):
    if text.endswith(",)"):
        text = text[2:-3]
    else:
        text = text[2:-2]
    return tuple(text.split("', '"))


def get_prompt_specific_terms(csv_file, n=150):
    table: list[tuple] = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        reader.__next__()

        for row in reader:
            table.append(tuple(row))

    sorted_terms = []
    for cluster in range(2, len(table[0])):
        for term in [eval(row[0]) for row in sorted(table, key=lambda x: float(x[cluster]), reverse=True)][:n]:
            sorted_terms += list(term)
    return set(sorted_terms)


def generate_modified_texts(important_words: List[set[str]], preprocess: Callable[[str], str] = None, input_folder="clusters",
                            output_folder="modified_clusters", inverted=False, verbose=False, replace_with_postag=True):
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        subprocess.call(["rm", "-r", output_folder])
        os.mkdir(output_folder)
    for i in range(15):  # iterate throw clusters
        os.mkdir("/".join([output_folder, str(i)]))
        files = os.listdir(input_folder + "/" + str(i))
        for i_file, file in enumerate(files):  # iterate throw all files
            if verbose:
                print("modifying cluster:", i, "-", str(i_file) + "/" + str(len(files)))
            if not os.path.isfile("/".join([input_folder, str(i), file])):  # filter non-files
                continue
            raw_text = data_loader.get_file("/".join([input_folder, str(i), file]))
            tokens = nltk.tokenize.word_tokenize(raw_text)
            tagged_tokens = nltk.pos_tag(tokens)
            for i_token, tt in enumerate(tagged_tokens):
                token, tag = tt
                post = token
                if preprocess is not None:
                    post = preprocess(post)
                if not xor((important_words[i].__contains__(post) and not closed_class_ptags.__contains__(tag)), inverted):  # filter token
                    if replace_with_postag:
                        tokens[i_token] = "["+tag+"]"
                    else:
                        tokens[i_token] = "dummy"
                if i_token % 15 == 14:
                    tokens[i_token] += "\n"

            output = open("/".join([output_folder, str(i), file]), "w")
            output.write(" ".join(tokens))


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("Starting text_modifier.py", start)

    n = 1000

    csv_file = "eval_1_output.csv"
    print("loading", csv_file)
    words = get_prompt_specific_terms(csv_file, n)

    print("")
    print("generate", "modified_clusters" + str(n))
    generate_modified_texts(
        words,
        preprocess=preprocessing.compose(preprocessing.lower, preprocessing.remove_quotes, preprocessing.remove_punctuation),
        output_folder="modified_clusters" + str(n)
    )
    print("")
    print("generate", "modified_clusters" + str(n) + "_inv")
    generate_modified_texts(
        words,
        preprocess=preprocessing.compose(preprocessing.lower, preprocessing.remove_quotes, preprocessing.remove_punctuation),
        output_folder="modified_clusters" + str(n) + "_inv",
        inverted=True
    )

    print("Done!")
    print("Time:", datetime.datetime.now()-start)
