def read_file(f):
    """Read file in CoNLL02 format

    Parameters
    ----------
    f : {file-like, string}
        Input stream or path.

    Returns
    -------
    sents : iterable over list
        Sentences, represented as lists of tokens.
    """
    if not hasattr(f, 'read'):
        f = open(f)

    sentence = []
    for ln in f:
        try:
            token, pos, netag = triple = ln.split()
            sentence.append(triple)
        except ValueError:
            yield sentence
            sentence = []

    if ln.strip() != '':
        yield sentence
