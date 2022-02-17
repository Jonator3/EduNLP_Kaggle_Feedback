import functools


def compose(*functions):
    """
    Composes an arbitrary number of functions.\n
    Given two functions f and g for example, compose returns a function
        h = g ∘ f
    such that
        h(args) = g(f(args)

    :return: fn ∘ ... ∘ f2 ∘ f1
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def remove_quotes(text: str) -> str:
    """
    Removes quotes from the given input text and returns the result.

    :param text: The input text that will be preprocessed
    :return: The input text without quotation-marks
    """
    return replace_with_nothing(text, ["`", '"', "¨", "'", "`", "´"])


def lower(text: str) -> str:
    """
    Converts all uppercase characters from text into lowercase characters and returns it.

    :param text: The input text that will be preprocessed
    :return: The lowercase input string
    """
    return text.lower()


