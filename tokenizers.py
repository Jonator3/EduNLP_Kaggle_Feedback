import nltk


def unigram_tokenize(text: str) -> list:
    return nltk.tokenize.word_tokenize(text)


def bigram_tokenize(text: str) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    return [(tokens[i], tokens[i + 1]) for i in range(max(len(tokens) - 1, 0))]


def trigram_tokenize(text: str) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(max(len(tokens) - 2, 0))]


def ngram_tokenize(text: str, n: int) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    ngrams = []
    for i in range(max(len(tokens) - (n-1), 0)):
        ngram = tuple([tokens[i + j] for j in range(n)])
        ngrams.append(ngram)

    return ngrams
