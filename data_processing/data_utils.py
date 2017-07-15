# coding=utf-8
import os
import cPickle as pkl
import geohash as geo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 由geohash获取经纬度
def get_lat(geohash):
    x,y,x_s,y_s = geo.decode_exactly(geohash)
    return x
def get_lon(geohash):
    x,y,x_s,y_s = geo.decode_exactly(geohash)
    return y

# 根据经纬度计算距离
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


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
# print( mapk( [ [3, 4, 5],[3, 4, 7] ],#(1/1+2/2) / 3 = 0.6
#             [[4, 7,1],[5, 7, 3]]  #(1/2) / 3 = 0.1
#              )  #( 0.6+0.1 ) / 2 = 0.35
#        )

def make_train_test(  ):
    tr = pd.read_csv('../data/train.csv')
    te = pd.read_csv('../data/test.csv')

    # handle exception point
    loc_cnt = tr[['geohashed_start_loc', 'orderid']].groupby('geohashed_start_loc', as_index=False).count()
    print('raw loc cnt : {0}'.format(len(loc_cnt)))
    loc_cnt = loc_cnt[loc_cnt.orderid > 5]
    print('final loc cnt : {0}'.format(len(loc_cnt)))
    loc_cnt = loc_cnt[['geohashed_start_loc']]

    tr = pd.merge(tr, loc_cnt, on='geohashed_start_loc')
    te = pd.merge(te, loc_cnt, on='geohashed_start_loc')

    # dig time metadata
    tr['datetime'] = pd.to_datetime(tr['starttime'])
    tr['weekday'] = tr['datetime'].apply(lambda dt: dt.weekday())
    tr['hour'] = tr['datetime'].apply(lambda dt: dt.hour)
    tr['quarter'] = tr['datetime'].apply(lambda dt: dt.hour * 4 + dt.minute / 15)
    del tr['datetime']
    pd.to_pickle(tr, '../data/tr_feat.pkl')
    print('trainset cnt :{0}'.format(len(tr)))

    te['datetime'] = pd.to_datetime(te['starttime'])
    te['weekday'] = te['datetime'].apply(lambda dt: dt.weekday())
    te['hour'] = te['datetime'].apply(lambda dt: dt.hour)
    te['quarter'] = te['datetime'].apply(lambda dt: dt.hour * 4 + dt.minute / 15)
    del te['datetime']

    # handle new user of testset
    feat = ['weekday', 'hour', 'geohashed_start_loc', 'userid']
    res = pd.DataFrame()
    res['userid'] = list(set(te['userid'].unique()) - set(tr['userid'].unique()))

    df = pd.merge(res, te[feat], on='userid')
    df = pd.merge(df, tr[feat], on=('weekday', 'hour', 'geohashed_start_loc'), how='left').fillna(-1)[
        ['userid_x', 'userid_y']].drop_duplicates('userid_x')
    df['userid_y'] = df['userid_y'].astype(int)
    df['is_new_user'] = 1
    df.columns = ['userid', 'old_user', 'is_new_user']

    te = pd.merge(te, df, on='userid', how='left').fillna(-1)
    te.loc[te.is_new_user == 1, ['userid']] = te.loc[te.is_new_user == 1, ['old_user']].values
    del te['old_user']
    del te['is_new_user']

    pd.to_pickle(te, '../data/te_feat.pkl')

if __name__ == '__main__':
    pass
