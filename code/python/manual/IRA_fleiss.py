# Inspired by https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75
"""
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

TABLE = 0
ROBOT = 1
TABLET = 2
ELSEWHERE = 3
UNKNOWN = 4

annotations_double = pd.read_csv('annotations_double_frame.csv')
annotations = pd.read_csv('annotations_frame.csv')

dfkappa = pd.DataFrame(columns=['file', 'case', 'ckappa'])
for f in annotations_double['file'].unique():
    fcase = list(annotations[annotations['file'] == f]['case'])[0]
    rater1 = list(annotations[annotations['file'] == f]['class'])
    rater2 = list(annotations_double[annotations_double['file'] == f]['class'])

    # rater1 = np.array(rater1).astype(int)
    # rater1[rater1==4] = 3
    # rater2 = np.array(rater2).astype(int)
    # rater2[rater2==4] = 3

    cohenk = cohen_kappa_score(rater1, rater2, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
    dfkappa.loc[len(dfkappa)] = [f, fcase, cohenk]

print(dfkappa['ckappa'].mean())
dfkappa.to_csv('ckappa_scores.csv')"""


# Adapted from https://gist.github.com/ShinNoNoir/4749548
def fleiss_kappa(ratings, n):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.

    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
        k: number of categories
    Returns:
        the Fleiss' kappa score

    See also:
        http://en.wikipedia.org/wiki/Fleiss'_kappa
    '''
    frames = set()
    classes = set()
    n_ij = {}

    for i, c in ratings:
        frames.add(i)
        classes.add(c)
        n_ij[(i, c)] = n_ij.get((i, c), 0) + 1

    N = len(frames)

    p_j = dict(((c, sum(n_ij.get((i, c), 0) for i in frames) / (1.0 * n * N)) for c in classes))
    P_i = dict(((i, (sum(n_ij.get((i, c), 0) ** 2 for c in classes) - n) / (n * (n - 1.0))) for i in frames))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(value ** 2 for value in p_j.values())

    f_kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return f_kappa


ratings = [(1, 'yes')] * 10 + [(1, 'no')] * 0 + [(2, 'yes')] * 8 + [(2, 'no')] * 2 + \
          [(3, 'yes')] * 9 + [(3, 'no')] * 1 + [(4, 'yes')] * 0 + [(4, 'no')] * 10 + [(5, 'yes')] * 7 + [(5, 'no')] * 3

print(ratings)