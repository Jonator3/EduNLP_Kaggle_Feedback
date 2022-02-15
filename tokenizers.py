import nltk


def unigram_tokenize(text: str) -> list:
    return nltk.tokenize.word_tokenize(text)


def bigram_tokenize(text: str) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = list(filter(lambda x: len(x) > 1, tokens))
    return [(tokens[i], tokens[i + 1]) for i in range(max(len(tokens) - 1, 0))]


def trigram_tokenize(text: str) -> list:
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = list(filter(lambda x: len(x) > 1, tokens))
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(max(len(tokens) - 2, 0))]