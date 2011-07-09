import logging
import numpy as np
from scikits.learn.grid_search import GridSearchCV
from scikits.learn.svm.sparse import LinearSVC
import scipy.sparse as sp

import conll
from features import extract_features
from util import bio_int


logger = logging.getLogger()


def train(sentences):
    """Train NER tagger.

    Parameters
    ----------
    sentences : iterable over list
        A sequence of lists of tokens.
    """
    if not isinstance(sentences, list):
        sentences = list(sentences)

    logger.debug("Extracting features")

    vocabulary = dict((t[0], i) for s in sentences for i, t in enumerate(s))

    X = []
    for i, s in enumerate(sentences):
        X.append(extract_features(s, vocabulary))
    X = sp.vstack(X, format='csr')

    # FIXME Only BIO tags for now
    y = np.array([bio_int[tok[2][0]] for s in sentences for tok in s])

    params = {
        "loss": ["l1", "l2"],
        "multi_class": [True, False],
        "C": [1., 10., 100.],
    }
    logger.debug("Training linear SVMs")
    clf = GridSearchCV(LinearSVC(), params, n_jobs=-1).fit(X, y)
    logger.debug("Done, returning the best one")
    return (clf.best_estimator, vocabulary)


if __name__ == "__main__":
    # Write pickled classifier to stdout.

    import cPickle as pickle
    import sys

    logging.basicConfig(level=logging.DEBUG)
    pickle.dump(train(conll.read_file(sys.argv[1])), sys.stdout)
