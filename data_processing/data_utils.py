# coding=utf-8
import pandas as pd

def split_train_test( split_day = '2017-05-18' ):
    tr = pd.read_csv('../data/train.csv')
    train,test = tr[tr.starttime<split_day],tr[tr.starttime>=split_day]
    return train,test


# tr,te = split_train_test()
# print len(tr) #1830100
# print len(te) #1383996

import numpy as np

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
print( mapk( [ [3, 4, 5],[3, 4, 7] ],#(1/1+2/2) / 3 = 0.6
            [[4, 7,1],[5, 7, 3]]  #(1/2) / 3 = 0.1
             )  #( 0.6+0.1 ) / 2 = 0.35
       )
