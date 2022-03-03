import csv
import datetime
from typing import Callable

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


def generate_modified_texts(important_words: set[str], preprocess: Callable[[str], str] = None, input_folder="clusters",
                            output_folder="modified_clusters"):
    os.mkdir(output_folder)
    for i in range(15):  # iterate throw clusters
        os.mkdir("/".join([output_folder, str(i)]))
        files = os.listdir(input_folder + "/" + str(i))
        for i_file, file in enumerate(files):  # iterate throw all files
            print("modifying cluster:", i, "-", str(i_file) + "/" + str(len(files)))
            if not os.path.isfile("/".join([input_folder, str(i), file])):  # filter non-files
                continue
            raw_text = data_loader.get_file("/".join([input_folder, str(i), file]))
            tokens = nltk.tokenize.word_tokenize(raw_text)
            for i_token, token in enumerate(tokens):
                post = token
                if preprocess is not None:
                    post = preprocess(post)
                if not (important_words.__contains__(post) or post == ""):  # filter token
                    tokens[i_token] = "dummy"

            output = open("/".join([output_folder, str(i), file]), "w")
            output.write(" ".join(tokens))


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("Starting text_modifier.py", start)

    n = 500

    csv_file = "eval_1_output.csv"
    print("loading", csv_file)
    words = get_prompt_specific_terms(csv_file, n)

    print("")
    generate_modified_texts(
        words,
        preprocess=preprocessing.compose(preprocessing.lower, preprocessing.remove_quotes, preprocessing.remove_punctuation),
        output_folder="modified_clusters" + str(n)
    )

    print("Done!")
    print("Time:", datetime.datetime.now()-start)
