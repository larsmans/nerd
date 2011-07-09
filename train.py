import numpy as np
from scikits.learn.grid_search import GridSearchCV
from scikits.learn.svm import LinearSVC
import logging

import conll
from features import extract_features, n_features
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
    #X = np.empty((sum(len(s) for s in sentences), n_features), dtype=bool)
    X = []
    for i, s in enumerate(sentences):
        #X[i] = extract_features(s)
        X.append(extract_features(s))
    X = np.concatenate(X)

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
    return clf.best_estimator


if __name__ == "__main__":
    # Write pickled classifier to stdout.

    import cPickle as pickle
    import sys

    logging.basicConfig(level=logging.DEBUG)
    pickle.dump(train(conll.read_file(sys.argv[1])), sys.stdout)
