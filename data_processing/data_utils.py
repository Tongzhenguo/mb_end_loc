# coding=utf-8
import geohash as geo
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def make_label():
    tr = pd.read_csv('../data/train.csv')
    te = pd.read_csv('../data/test.csv')

    loc_raw = set(tr['geohashed_start_loc'].values).union(set(te['geohashed_start_loc'].values).union( set(tr['geohashed_end_loc'].values) ))
    print len(loc_raw) #110527

    print len( set([loc[:6] for loc in loc_raw ] ) ) #8571

    loc_array = np.array( list(set([loc[:6] for loc in loc_raw])) ).reshape([-1,1])
    le = LabelEncoder()
    le = le.fit( loc_array )
    tr['geohashed_start_loc_enc'] = le.transform( tr['geohashed_start_loc'].apply(lambda x:x[:6]).values )
    tr['geohashed_end_loc_enc'] = le.transform(tr['geohashed_end_loc'].apply(lambda x:x[:6]).values)

    te['geohashed_start_loc_enc'] = le.transform(te['geohashed_start_loc'].apply(lambda x:x[:6]).values)
    print tr.head()
    return tr,te


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


def do_bad():
    tr = pd.read_csv('../data/train.csv')
    te = pd.read_csv('../data/test.csv')

    tr['f2'] = tr['geohashed_start_loc'].apply(lambda x: x[:2])
    tr['f2'] = tr['geohashed_end_loc'].apply(lambda x: x[:2])
    te['f2'] = te['geohashed_start_loc'].apply(lambda x: x[:2])

    black_list = ['ws','wt','ww']
    tr = tr[ (tr.f2.isin(black_list)) ]
    te = te[(te.f2.isin(black_list)) ]

    res = pd.DataFrame()
    loc_times = tr.groupby(['geohashed_end_loc'], as_index=False).count()[['geohashed_end_loc', 'orderid']]
    loc_times.columns = ['loc', 'times']
    res = res.append(loc_times)

    loc_times = tr.groupby(['geohashed_start_loc'], as_index=False).count()[['geohashed_start_loc', 'orderid']]
    loc_times.columns = ['loc', 'times']
    res = res.append(loc_times)

    loc_times = te.groupby(['geohashed_start_loc'], as_index=False).count()[['geohashed_start_loc', 'orderid']]
    loc_times.columns = ['loc', 'times']
    res = res.append(loc_times)

    res = res.groupby(['loc'], as_index=False).sum()
    res['f2'] = res['loc'].apply(lambda x: x[:2])
    res['rn'] = res['times'].groupby(res['f2']).rank(axis=0, ascending=False, method='dense')
    hot3 = res[res.rn <= 3.0].drop_duplicates(['f2', 'rn'])

    res1 = hot3[hot3.rn == 1.0][['loc', 'f2']]
    res1.columns = ['hot1', 'f2']
    res2 = hot3[hot3.rn == 2.0][['loc', 'f2']]
    res2.columns = ['hot2', 'f2']
    res3 = hot3[hot3.rn == 3.0][['loc', 'f2']]
    res3.columns = ['hot3', 'f2']

    res = pd.merge(res, res1, on='f2', how='left')[['loc', 'hot1', 'f2']]
    res = pd.merge(res, res2, on='f2', how='left').fillna('-1')[['loc', 'hot1', 'hot2', 'f2']]
    res = pd.merge(res, res3, on='f2', how='left').fillna('-1')[['loc', 'hot1', 'hot2', 'hot3']]
    res.to_csv('../data/loc_bad_hot3_loc.csv',index=False)
    return res

# print do_bad().head()

#计算一个地点内其邻近的三个热点位置
def loc_loc(  ):
    tr = pd.read_csv('../data/train.csv')
    te = pd.read_csv('../data/test.csv')

    res = pd.DataFrame()
    loc_times = tr.groupby(['geohashed_end_loc'],as_index=False).count()[['geohashed_end_loc','orderid']]
    loc_times.columns = ['loc','times']
    res = res.append( loc_times )

    loc_times = tr.groupby(['geohashed_start_loc'], as_index=False).count()[['geohashed_start_loc', 'orderid']]
    loc_times.columns = ['loc', 'times']
    res = res.append(loc_times)

    loc_times = te.groupby(['geohashed_start_loc'], as_index=False).count()[['geohashed_start_loc', 'orderid']]
    loc_times.columns = ['loc', 'times']
    res = res.append(loc_times)
    res = res.groupby(['loc'],as_index=False).sum()


    res['f5'] = res['loc'].apply( lambda x:x[:5] )
    res = res[ res.f5.str.startswith('wx') ]
    print 'test dateset all loc num:', len(res)

    res['rn'] = res['times'].groupby( res['f5'] ).rank(axis=0,ascending=False,method='dense')
    hot3 = res[res.rn<=3.0].drop_duplicates(['f5','rn'])

    res1 = hot3[hot3.rn==1.0][['loc','f5']]
    res1.columns = ['hot1','f5']
    res2 = hot3[hot3.rn == 2.0][['loc','f5']]
    res2.columns = ['hot2', 'f5']
    res3 = hot3[hot3.rn == 3.0][['loc','f5']]
    res3.columns = ['hot3', 'f5']

    res = pd.merge( res,res1,on='f5',how='left')[['loc','hot1','f5']]
    res = pd.merge(res, res2, on='f5',how='left').fillna('-1')[['loc','hot1','hot2','f5']]
    res = pd.merge(res, res3, on='f5',how='left').fillna('-1')[['loc','hot1','hot2','hot3']]

    print len(res)
    res.to_csv('../data/loc_hot3_loc.csv',index=False)
    return res
# print loc_loc().head(40)

def split_train_test( split_day = '2017-05-21' ):
    tr = pd.read_csv('../data/train.csv')
    train,test = tr[tr.starttime<split_day],tr[tr.starttime>=split_day]
    return train,test

#
# tr,te = split_train_test()
# print len(tr) #1830100
# print len(te) #1383996
#1367872
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
# print( mapk( [ [3, 4, 5],[3, 4, 7] ],#(1/1+2/2) / 3 = 0.6
#             [[4, 7,1],[5, 7, 3]]  #(1/2) / 3 = 0.1
#              )  #( 0.6+0.1 ) / 2 = 0.35
#        )
