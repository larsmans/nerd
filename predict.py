from features import extract_features
from train import train
from util import int_bio


def predict(clf, sentence):
    """Predict BIO labels for a single sentence."""

    X = extract_features(sentence)
    pred = [int_bio[y] for y in clf.predict(X)]

    # Heuristic repair: make output consistent,
    # but never worse than the raw prediction.
    for i in xrange(len(pred)):
        if pred[i] == "I" and (i == 0 or pred[i - 1] == "O"):
            pred[i] = "B"

    return pred


if __name__ == "__main__":
    from conll import read_file
    import cPickle as pickle
    import sys

    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: %s clf input_file" % sys.argv[0]
        sys.exit(1)

    clf = pickle.load(open(sys.argv[1]))

    for sentence in read_file(sys.argv[2]):
        Y_pred = predict(clf, sentence)
        for (token, pos, y_true), y_pred in zip(sentence, Y_pred):
            print token, pos, y_true[0], y_pred
        print
