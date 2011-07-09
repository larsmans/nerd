import numpy as np
import re


def extract_features(sentence):
    """Do feature extraction on a single sentence.

    We need a sentence, rather than a token, since some features depend
    on the context of tokens.
    """
    n_tokens = len(sentence)
    X = np.empty((n_tokens, n_features), dtype=bool)

    for i in xrange(n_tokens):
        for j, f in enumerate(FEATURES):
            X[i, j] = f(sentence, i)

    return X


# Spelling
def first_of_sentence(s, i):
    return i == 0

def all_caps(s, i):
    return s[i][0].isupper()

def initial_cap(s, i):
    return s[i][0][0].isupper()

def has_dash(s, i):
    return '-' in s[i]

def has_num(s, i):
    return re.search(r"[0-9]", s[i]) is not None

# FIXME Learn POS tags from training data.
def isadj(s, i):
    return s[i][1] == "Adj"

def isadv(s, i):
    return s[i][1] == "Adv"

def isart(s, i):
    return s[i][1] == "Art"

def isconj(s, i):
    return s[i][1] == "Conj"

def isint(s, i):
    return s[i][1] == "Int"

def ismisc(s, i):
    return s[i][1] == "Misc"

def isnoun(s, i):
    return s[i][1] == "N"

def isnum(s, i):
    return s[i][1] == "Num"

def isprep(s, i):
    return s[i][1] == "Prep"

def ispunc(s, i):
    return s[i][1] == "Punc"

def ispron(s, i):
    return s[i][1] == "Pron"

def isverb(s, i):
    return s[i][1] == "V"

# Feature metafunctions
def conj(fs):
    """Conjunction of features fs"""
    def feature(s, i):
        return all(f(s, i) for f in fs)
    return feature

def butnot(f1, f2):
    def feature(s, i):
        return f1(s, i) and not f2(s, i)
    return feature

def nextf(f, offset=1):
    """Next token has feature f"""
    def feature(s, i):
        i += offset
        return i < len(s) and f(s, i)
    return feature

def prevf(f, offset=1):
    """Previous token has feature f"""
    def feature(s, i):
        i -= offset
        return i >= 0 and f(s, i)
    return feature


FEATURES = [initial_cap, all_caps, first_of_sentence, has_dash,
            #has_num,
            #butnot(initial_cap, first_of_sentence),
            prevf(initial_cap), prevf(all_caps),
            isadj, isadv, isart, isconj, isint, ismisc,
            isnoun, isnum, isprep, ispunc, ispron, isverb]
n_features = len(FEATURES)


if __name__ == '__main__':
    import conll
    import sys

    for s in conll.read_file(sys.argv[1]):
        for i in xrange(len(s)):
            print ' '.join(s[i]),
            for f in extract_features(s):
                print '%s:%d' % f,
        print
