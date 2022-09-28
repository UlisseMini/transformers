import string
import sys

alphanumeric = string.ascii_letters + string.digits

def get_latex_sentences(text):
    # TODO: wrap sentences in analysis.txt
    return [list(latex(s)) for s in text.split('\n')]


def get_char_sentences(data, sep='\n'):
    words = data.split(sep)
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    return words

def tokenize_word(text, i):
    """
    >>> tokenize_word("word", 0)
    ('word', 4)
    >>> tokenize_word("word!", 0)
    ('word', 4)
    >>> tokenize_word("foo bar!", 0)
    ('foo', 3)
    """

    s = i
    while i < len(text) and text[i] in alphanumeric:
        i += 1
    return text[s:i], i



def latex(text):
    r"""
    Tokenize a string of LaTeX code, with math.
    >>> list(latex(r"word"))
    ['word']
    >>> list(latex(r"word."))
    ['word', '.']
    >>> list(latex(r"\texword"))
    ['\\texword']
    >>> list(latex(r"\section{Epic}."))
    ['\\section', '{', 'Epic', '}', '.']
    >>> list(latex(r"Hello, \textit{world}!"))
    ['Hello', ',', ' ', '\\textit', '{', 'world', '}', '!']
    >>> list(latex(r"$2+2=4$ quick maths!"))
    ['$', '2', '+', '2', '=', '4', '$', ' ', 'quick', ' ', 'maths', '!']
    """

    i = 0
    while i < len(text):
        if text[i:i+2] == '$$':
            yield '$$'
        elif text[i] == '$':
            yield '$'
        elif text[i:i+2] == '\\[' or text[i:i+2] == '\\]':
            yield '$$'
        elif text[i:i+2] == '\\(' or text[i:i+2] == '\\)':
            yield '$'
        elif text[i:i+2] == '\\\\':
            yield '\\\\'
        elif text[i] == '\\':
            tok, j = tokenize_word(text, i+1)
            i = j
            yield '\\' + tok
            continue
        # number
        elif text[i] in string.digits:
            # we yield digits of numbers, otherwise math would be hard for transformer
            yield text[i]
        # word
        elif text[i] in string.ascii_letters:
            tok, j = tokenize_word(text, i)
            i = j
            yield tok
            continue
        # general cases
        elif text[i] in string.punctuation:
            yield text[i]
        elif text[i] in string.whitespace:
            yield text[i]
        
        i += 1

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print('done')