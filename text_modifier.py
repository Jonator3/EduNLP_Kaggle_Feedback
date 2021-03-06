import csv
import datetime
from argparse import ArgumentParser
from typing import Callable, List
from operator import xor
import subprocess

import nltk
import os

import data_loader
import preprocessing


closed_class_ptags = ["Cd", "CC", "DT", "EX", "IN", "LS", "MD", "PDT", "POS",
                      "PRP", "PRP$", "RP", "TO", "UH", "WDT", "WP", "WP$", "WRB"]


def parse_ngram(text: str):
    if text.endswith(",)"):
        text = text[2:-3]
    else:
        text = text[2:-2]
    return tuple(text.split("', '"))


def get_prompt_specific_terms(csv_file, n=150) -> List[set[str]]:
    table: list[tuple] = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        reader.__next__()

        for row in reader:
            table.append(tuple(row))

    sorted_terms = []
    for cluster in range(2, len(table[0])):
        sorted_terms.append([eval(row[0]) for row in sorted(table, key=lambda x: float(x[cluster]), reverse=True)][:n])

    output = []
    for terms in sorted_terms:
        l = []
        for term in terms:
            for word in term:
                l.append(word)
        output.append(set(l))

    return output


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
    arg_pars = ArgumentParser()
    arg_pars.add_argument("input_file", help="The path to the directory containing the clusters", default=None)
    arg_pars.add_argument("boarder-position", help="the boarder at witch to separate the tokens in the output", type=int, default=None)
    arg_pars.add_argument("--output", help="full path used for the output", default=None)
    args = arg_pars.parse_args()

    csv_file = args.input_folder  # path to the input clusters
    if csv_file is None:
        csv_file = input("Enter the path to the input file:\n")
        print("")

    n = args.boarder_position
    if n is None:
        n = input("Enter boarder position:\n")
        print("")
        while not n.isdigit():
            n = input("Input must be an Integer!\nEnter n-gram length:\n")
            print("")
        n = int(n)
    
    output_path = args.output
    if output_path is None:
        output_path = "modified_clusters" + str(n)
    
    start = datetime.datetime.now()
    print("Starting text_modifier.py", start)

    print("loading", csv_file)
    words = get_prompt_specific_terms(csv_file, n)

    print("")
    print("generate", output_path)
    generate_modified_texts(
        words,
        preprocess=preprocessing.compose(preprocessing.lower, preprocessing.remove_quotes, preprocessing.remove_punctuation),
        output_folder=output_path
    )
    print("")
    print("generate", output_path + "_inv")
    generate_modified_texts(
        words,
        preprocess=preprocessing.compose(preprocessing.lower, preprocessing.remove_quotes, preprocessing.remove_punctuation),
        output_folder=output_path + "_inv",
        inverted=True
    )

    print("Done!")
    print("Time:", datetime.datetime.now()-start)
