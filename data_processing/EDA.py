# coding=utf-8
import math
import numpy as np
import pandas as pd
from scipy.stats import vonmises_gen
import scipy.special as sc
import matplotlib.pyplot as plt
tr = pd.read_csv('../data/train.csv')
te = pd.read_csv('../data/test.csv')

# print(len(tr))  # 3214096
# print(len(tr['userid'].unique()))  # 349693
# print(len(tr['bikeid'].unique()))  # 485465
# print(tr['biketype'].unique())  # [1,2]
# print('time uniq',tr['starttime'].unique().shape[0] )
#
# df = pd.to_datetime( tr['starttime'] )
# print(df.min())  # 2017-05-10 00:00:09
# print(df.max())  # Timestamp('2017-05-24 19:00:16')
#
# print(len(list(tr['geohashed_start_loc'].unique()) + list(tr['geohashed_end_loc'].unique())))  # 178934
# start_end_pairs =  tr[['geohashed_start_loc','geohashed_end_loc','orderid']].groupby(
#      ['geohashed_start_loc','geohashed_end_loc'],as_index=False
#  ).count().sort_values(['orderid'],ascending=False)
# start_end_pairs.columns = ['geohashed_start_loc','geohashed_end_loc','times'] ##1420276
# start_end_pairs[start_end_pairs.times>10]['times'].hist() # 频繁项长尾分布：大部分在行程路线只出现不到78次，最频繁行程有681次
# plt.show()
# print(start_end_pairs.head(10))
# '''
#        geohashed_start_loc geohashed_end_loc  times
# 696727             wx4f9ky           wx4f9mk    681
# 697517             wx4f9mk           wx4f9ky    497
# 696393             wx4f9kn           wx4f9mk    437
# 696395             wx4f9kn           wx4f9ms    372
# 867271             wx4fg87           wx4ferq    356
# 696729             wx4f9ky           wx4f9ms    356
# 708646             wx4f9wb           wx4f9mu    355
# 697694             wx4f9ms           wx4f9kn    345
# 416470             wx4eq0c           wx4eq23    323
# 697507             wx4f9mk           wx4f9kn    319
# '''

#没有在训练集中出现的用户，地点
# print('new users cnt :',len( set(te['userid'].unique()) - set(tr['userid'].unique()) )) #715893
# print('new loc cnt :',len( set(te['geohashed_start_loc'].unique()) - set(tr['geohashed_start_loc'].unique()) )) #10117

def loc_desc(  ):
    global tr
    #x = geohashed_end_loc | geohashed_start_loc 满足多项分布
    a = tr[['geohashed_start_loc','orderid']].groupby( 'geohashed_start_loc',as_index=False ).count()
    b = tr[['geohashed_start_loc','geohashed_end_loc','orderid']].groupby( ['geohashed_start_loc','geohashed_end_loc'],as_index=False ).count()
    c = pd.merge( a,b,on='geohashed_start_loc' )
    c['prob'] = 1.0 * c['orderid_y'] / c['orderid_x']
    c = c[['geohashed_start_loc','geohashed_end_loc','prob']]
    c.to_csv('../data/start_end_desc.csv')

def time_desc():
    df = tr[['weekday', 'hour', 'quarter', 'geohashed_end_loc', 'orderid']].groupby(
        ['weekday', 'hour', 'quarter', 'geohashed_end_loc'], as_index=False).count()
    a = df.groupby(['weekday', 'hour', 'quarter'], as_index=False).sum()[['weekday', 'hour', 'quarter', 'orderid']]
    b = pd.merge(df, a, on=('weekday', 'hour', 'quarter'))
    b['prob'] = 1.0 * b['orderid_x'] / b['orderid_y']
    b = b[['weekday', 'hour', 'quarter','prob']]
    b.to_csv('../data/time_end_desc.csv')

def hour_desc():
    d = pd.read_csv('../data/data.csv')
    def fn( x ):
        a = list(x)
        a.sort()
        a = map(str,a)
        return '/'.join(a)
    dd = d[['userid','geohashed_end_loc','hour']].groupby(['userid','geohashed_end_loc'])['hour'].apply(fn).reset_index()
    print dd.head()
    pd.to_pickle(dd,'../data/hour_desc.pkl')
    return dd
# hour_desc()
# d = pd.read_csv('../data/data.csv')
# df = d[(d.userid==957398) & (d.geohashed_end_loc=='wx4gp87')].sort_values('hour')[['hour']].sort_values('hour')
# x = df['hour'].values
# n, bins, patches = plt.hist(x, df['hour'].unique().shape[0], facecolor='g', alpha=0.75)
#
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# # 添加标题
# plt.title('Histogram of hour')
# # 添加文字
# plt.axis([0, 23, 0, 20])
# plt.grid(True)
# plt.show()

def cal_mu( xs ):
    """
    :param xs: ndarray
    :return:
    """
    return round(np.mean( xs ),1)
def cal_sigma( xs,u ):
    """
    :param xs:
    :return:
    """
    if not u:
        u = cal_mu( xs )
    def G( xs,u ):
        return np.sum( (-abs(abs( xs - u ) - 12)+12) ** 2 )
    return float(1) / len(xs) * G( xs,u )
vonmises_hour = vonmises_gen(a=0, b=24, name='vonmises_hour')
def get_hour_prob( hours,u,sigma ):
    # y = [ (h-12)/12.0*math.pi for h in hours ]
    # u =(u-12) / 12.0*math.pi
    # sigma = sigma / 12.0 *math.pi
    # print( y )
    # print( "均值={u},sigma={sigma}".format(u=u,sigma=sigma) )
    return vonmises_hour.pdf(hours,1.0/sigma , loc=u, scale=math.sqrt(sigma) )
print get_hour_prob([9,],10.5,2.0)
print get_hour_prob([9,],19.3,0.96)
print get_hour_prob([9,],19.5,0.25)
print get_hour_prob([9,],19,0.33)
print get_hour_prob([9,],22.5,0.25)
# d = pd.read_csv('../data/data.csv')
# df = d[(d.userid==957398) & (d.geohashed_end_loc=='wx4gp87')]
# u = cal_mu( df['hour'].values )
# k = cal_sigma( df['hour'].values,u )
# print( df['hour'].values )
# print( get_hour_prob( df['hour'].values,k,u,24 ) )
# print( pd.read_csv('../data/hour_rate.csv') )

# if __name__ == "__main__":
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#
#     df = pd.read_csv('../data/hour_rate.csv')
#     df.plot()
#     plt.show()
