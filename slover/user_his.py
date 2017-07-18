# coding=utf-8
import pandas as pd

tr = pd.read_csv('../data/train.csv')

# dig time metadata
def trans_time( tr ):
    tr['datetime'] = pd.to_datetime(tr['starttime'])
    tr['weekday'] = tr['datetime'].apply(lambda dt: dt.weekday())
    tr['hour'] = tr['datetime'].apply(lambda dt: dt.hour)
    def time_range( hh ):
        if hh in (6,7,8,9):
            return 8
        if hh in (17,18,19,20,21,22):
            return 17
        return hh
    tr['time_range'] = tr['hour'].apply( time_range )
    tr['quarter'] = tr['datetime'].apply(lambda dt: int( dt.hour * 4 + dt.minute / 15)+1 )
    del tr['datetime']
    print('trainset cnt :{0}'.format(len(tr)))

    df2 = tr.sort_values(['userid','starttime']).drop_duplicates()
    del df2['starttime']
    return df2

#增加取top3功能，多了没用
def get_user_hour_prob( df2 ):
    df2_hour_end = df2.groupby(['userid', 'time_range', 'geohashed_end_loc'], as_index=False).count()[
        ['userid', 'time_range', 'geohashed_end_loc', 'orderid']]
    df2_hour = df2.groupby(['userid', 'time_range'], as_index=False).count()[
        ['userid', 'time_range','orderid']]
    df = pd.merge( df2_hour_end,df2_hour,on=('userid', 'time_range') )
    df['prob'] = 1.0 * df['orderid_x'] / df['orderid_y']
    df = df[['userid', 'time_range', 'geohashed_end_loc', 'prob']]
    return df

def get_user_start_prob( df2 ):
    df2_start_end = df2.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False).count()[['userid','geohashed_start_loc','geohashed_end_loc','orderid']]
    df2_start = df2.groupby(['userid','geohashed_start_loc'],as_index=False).count()[['userid','geohashed_start_loc','orderid']]
    df = pd.merge( df2_start_end,df2_start,on=('userid','geohashed_start_loc') )
    df['prob'] = 1.0*df['orderid_x'] / df['orderid_y']
    df = df[['userid','geohashed_start_loc','geohashed_end_loc','prob']]
    return df

df2 = trans_time(tr)
df2.to_csv('../data/data.csv',index=None)
user_hour_prob = get_user_hour_prob( df2 )
user_start_prob = get_user_start_prob( df2 )

te = pd.read_csv('../data/test.csv')
te = trans_time( te )
res = pd.DataFrame()
res = res.append( pd.merge( te[['orderid','userid','time_range']],user_hour_prob,on=('userid','time_range') )[['orderid','geohashed_end_loc','prob']] )
res = res.append( pd.merge( te[['orderid','userid','geohashed_start_loc']],user_start_prob,on=('userid','geohashed_start_loc') )[['orderid','geohashed_end_loc','prob']] )
groupby = res.sort_values('prob', ascending=False).groupby('orderid', as_index=False)
n1 = groupby.nth(0)
n1.columns = ['orderid','pred1','prob1']
n2 = groupby.nth(1)
n2.columns = ['orderid','pred2','prob2']
n3 = groupby.nth(2)
n3.columns = ['orderid','pred3','prob3']
res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
res = res[['orderid', 'pred1', 'pred2', 'pred3']]


r = pd.read_csv('../res/syp_userid_loc.csv', names=['orderid', 'pred1', 'pred2', 'pred3'])
a = pd.DataFrame()
a['orderid'] = list(set(r['orderid'].unique()) - set(res['orderid'].unique()))

r = pd.merge(r, a, on='orderid')
res = res.append(r[['orderid', 'pred1', 'pred2', 'pred3']]).drop_duplicates('orderid')
res.to_csv('../res/res_user.csv', header=None, index=None)
print(len(res))




