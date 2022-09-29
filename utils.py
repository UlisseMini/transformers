import string
import sys
import pickle
import hashlib
from pathlib import Path
import itertools


alphanumeric = string.ascii_letters + string.digits

# https://github.com/joseluis9595/pickled-decorator/blob/master/pickled/pickled.py
# yes this is cursed
def pickled(func):
    def wrapper(*args, **kwargs):
        # Pickle arguments
        pickled_args = pickle.dumps({"args": args, "kwargs": kwargs})
        # Create md5 hash of the pickled object
        md5_hashed_args = hashlib.md5(str(pickled_args).encode('utf-8')).hexdigest()

        # Check existence of cached response
        pickle_filepath = Path(f"/tmp/pickled_functions/{md5_hashed_args}.pkl")
        pickle_filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"\033[96m*** Searching for cached response for '{func.__name__}()\033[0m'")
            result = pickle.load(open(pickle_filepath, "rb"))
            print(f"\033[92m*** Using cached response for '{func.__name__}()'\033[0m")
        except (OSError, IOError) as e:
            print(f"\033[93m*** No cache available. Executing '{func.__name__}()'\033[0m")
            result = func(*args, **kwargs)
            pickle.dump(result, open(pickle_filepath, "wb"))
        return result

    return wrapper


def join_sentences(sentences: list, n_sep_each: int, sep: str = '\n'):
    """
    >>> join_sentences(["a", "b", "c"], 1, " ")
    ['a', 'b', 'c']
    >>> join_sentences(["a", "b", "c"], 2, " ")
    ['a b', 'b c']
    >>> join_sentences(["a", "b", "c", "d"], 3, "-")
    ['a-b-c', 'b-c-d']
    """

    return [
        sep.join(itertools.chain(*sentences))
        for sentences in zip(*[sentences[k:] for k in range(n_sep_each)])
    ]

@pickled
def get_latex_sentences(text):
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