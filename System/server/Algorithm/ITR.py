from math import log2
import warnings


def itr(n, p, t):
    if (p < 0 or 1 < p):
        raise ValueError('stats:itr:BadInputValue ' \
                         + 'Accuracy need to be between 0 and 1.')
    elif (p < 1 / n):
        warnings.warn('stats:itr:BadInputValue ' \
                      + 'The ITR might be incorrect because the accuracy < chance level.')
        itr = 0
    elif (p == 1):
        itr = log2(n) * 60 / t
    else:
        itr = (log2(n) + p * log2(p) + (1 - p) * log2((1 - p) / (n - 1))) * 60 / t

    return itr
