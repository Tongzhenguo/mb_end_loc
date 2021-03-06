# coding=utf-8
import gc
import pandas as pd

'''
    新增用户终点起点反转的概率
'''

# dig time metadata
def trans_time( tr ):
    tr['datetime'] = pd.to_datetime(tr['starttime'])
    tr['weekday'] = tr['datetime'].apply(lambda dt: 1 if dt.weekday() <5 else 2) #Monday == 0 ... Sunday == 6
    tr['hour'] = tr['datetime'].apply(lambda dt: dt.hour)
    def time_range( hh ):
        if hh in (6,7,8,9):
            return 8
        if hh in (17,18,19,20,21,22):
            return 17
        if hh <6 or hh>22:
            return 7
        if hh>9 and hh<17:
            return 10
    tr['time_range'] = tr['hour'].apply( time_range )
    tr['quarter'] = tr['datetime'].apply(lambda dt: int( dt.hour * 4 + dt.minute / 15)+1 )
    del tr['datetime']
    print('trainset cnt :{0}'.format(len(tr)))

    df2 = tr.sort_values(['userid','starttime']).drop_duplicates()
    del df2['starttime']
    return df2

def get_user_quarter_prob( df2,a,te ):
    """
    :param df2:
    :param a:权重控制参数,0到1之间的浮点数
    :return:
    """
    df2_hour_end = df2.groupby(['userid', 'quarter', 'geohashed_end_loc'], as_index=False).count()[
        ['userid', 'quarter', 'geohashed_end_loc', 'orderid']]
    df2_hour = df2.groupby(['userid', 'quarter'], as_index=False).count()[
        ['userid', 'quarter','orderid']]
    df = pd.merge( df2_hour_end,df2_hour,on=('userid', 'quarter') )
    df['prob'] = a * df['orderid_x'] / df['orderid_y']
    df = df[['userid', 'quarter', 'geohashed_end_loc', 'prob']]

    df1 = pd.merge(te[['orderid', 'userid', 'quarter']], df, on=('userid', 'quarter'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    del df
    gc.collect()
    groupby = df1.sort_values('prob', ascending=False).drop_duplicates(['orderid', 'geohashed_end_loc']).groupby(
        'orderid', as_index=False)
    n1 = groupby.nth(0)
    n1.columns = ['orderid', 'pred1', 'prob1']
    n2 = groupby.nth(1)
    n2.columns = ['orderid', 'pred2', 'prob2']
    n3 = groupby.nth(2)
    n3.columns = ['orderid', 'pred3', 'prob3']
    res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
    res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
    del res['prob2']
    del res['prob3']
    return res


def get_user_weekday_prob( df2,a,te ):
    """
    :param df2:
    :param a:权重控制参数,0到1之间的浮点数
    :return:
    """
    df2_hour_end = df2.groupby(['userid', 'weekday', 'geohashed_end_loc'], as_index=False).count()[
        ['userid', 'weekday', 'geohashed_end_loc', 'orderid']]
    df2_hour = df2.groupby(['userid', 'weekday'], as_index=False).count()[
        ['userid', 'weekday','orderid']]
    df = pd.merge( df2_hour_end,df2_hour,on=('userid', 'weekday') )
    df['prob'] = a * df['orderid_x'] / df['orderid_y']
    df = df[['userid', 'weekday', 'geohashed_end_loc', 'prob']]

    df1 = pd.merge(te[['orderid', 'userid', 'weekday']], df, on=('userid', 'weekday'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    del df
    gc.collect()
    groupby = df1.sort_values('prob', ascending=False).drop_duplicates(['orderid', 'geohashed_end_loc']).groupby(
        'orderid', as_index=False)
    n1 = groupby.nth(0)
    n1.columns = ['orderid', 'pred1', 'prob1']
    n2 = groupby.nth(1)
    n2.columns = ['orderid', 'pred2', 'prob2']
    n3 = groupby.nth(2)
    n3.columns = ['orderid', 'pred3', 'prob3']
    res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
    res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
    del res['prob2']
    del res['prob3']
    return res


def get_user_hour_prob( df2,a,te ):
    """
    :param df2:
    :param a:权重控制参数,0到1之间的浮点数
    :return:
    """
    df2_hour_end = df2.groupby(['userid', 'time_range', 'geohashed_end_loc'], as_index=False).count()[
        ['userid', 'time_range', 'geohashed_end_loc', 'orderid']]
    df2_hour = df2.groupby(['userid', 'time_range'], as_index=False).count()[
        ['userid', 'time_range','orderid']]
    df = pd.merge( df2_hour_end,df2_hour,on=('userid', 'time_range') )
    df['prob'] = a * df['orderid_x'] / df['orderid_y']
    df = df[['userid', 'time_range', 'geohashed_end_loc', 'prob']]

    df1 = pd.merge(te[['orderid', 'userid', 'time_range']], df, on=('userid', 'time_range'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    del df
    gc.collect()
    groupby = df1.sort_values('prob', ascending=False).drop_duplicates(['orderid', 'geohashed_end_loc']).groupby(
        'orderid', as_index=False)
    n1 = groupby.nth(0)
    n1.columns = ['orderid', 'pred1', 'prob1']
    n2 = groupby.nth(1)
    n2.columns = ['orderid', 'pred2', 'prob2']
    n3 = groupby.nth(2)
    n3.columns = ['orderid', 'pred3', 'prob3']
    res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
    res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
    del res['prob2']
    del res['prob3']
    return res

def get_user_start_prob( df2,a,te ):
    """
    :param df2:
    :param a:权重控制参数,0到1之间的浮点数
    :return:
    """
    df2_start_end = df2.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False).count()[['userid','geohashed_start_loc','geohashed_end_loc','orderid']]
    df2_start = df2.groupby(['userid','geohashed_start_loc'],as_index=False).count()[['userid','geohashed_start_loc','orderid']]
    df = pd.merge( df2_start_end,df2_start,on=('userid','geohashed_start_loc') )
    df['prob'] = a*df['orderid_x'] / df['orderid_y']
    df = df[['userid','geohashed_start_loc','geohashed_end_loc','prob']]
    df2 = df
    df2['t'] = df2['geohashed_start_loc']
    df2['geohashed_start_loc'] = df2['geohashed_end_loc']
    df2['geohashed_end_loc'] = df2['t']
    del df2['t']
    df = df.append( df2 )
    del df2

    df1 = pd.merge(te[['orderid', 'userid', 'geohashed_start_loc']], df, on=('userid', 'geohashed_start_loc'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    del df
    gc.collect()
    groupby = df1.sort_values('prob', ascending=False).drop_duplicates(['orderid', 'geohashed_end_loc']).groupby(
        'orderid', as_index=False)
    n1 = groupby.nth(0)
    n1.columns = ['orderid', 'pred1', 'prob1']
    n2 = groupby.nth(1)
    n2.columns = ['orderid', 'pred2', 'prob2']
    n3 = groupby.nth(2)
    n3.columns = ['orderid', 'pred3', 'prob3']
    res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
    res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
    del res['prob2']
    del res['prob3']
    return res


def make_eval(val):
    user_hour_prob = get_user_hour_prob( val,1.0,val )
    user_start_prob = get_user_start_prob( val,1.0,val )
    user_weekday_prob = get_user_weekday_prob(val,1.0,val)
    user_quarter_prob = get_user_quarter_prob(val, 1.0,val)

    res = pd.DataFrame()
    res = res.append( user_hour_prob )
    res = res.append(user_start_prob)
    res = res.append(user_weekday_prob)
    res = res.append(user_quarter_prob)

    res.sort_values('prob1',ascending=False).drop_duplicates('orderid')
    res.to_pickle('../data/a.csv')

    # # res = pd.DataFrame()
    # # 这里假设时段和起点相互独立
    # df1 = pd.merge(val[['orderid', 'userid', 'time_range']], user_hour_prob, on=('userid', 'time_range'))[
    #     ['orderid', 'geohashed_end_loc', 'prob']]
    # df2 = pd.merge(val[['orderid', 'userid', 'geohashed_start_loc']], user_start_prob, on=('userid', 'geohashed_start_loc'))[['orderid', 'geohashed_end_loc', 'prob']]
    # df3 = pd.merge(val[['orderid', 'userid', 'weekday']], user_weekday_prob, on=('userid', 'weekday'))[
    #     ['orderid', 'geohashed_end_loc', 'prob']]
    # df4 = pd.merge(val[['orderid', 'userid', 'quarter']], user_quarter_prob, on=('userid', 'quarter'))[
    #     ['orderid', 'geohashed_end_loc', 'prob']]



    # res = pd.merge( df1,df2,on=('orderid','geohashed_end_loc'),how='left' ).fillna(0.001) #保证p(end|hour)*p(end|start)大于只在end|start中出现的概率，忽略了他们的差异
    # res['prob'] = res['prob_x']*res['prob_y']
    # del res['prob_x']
    # del res['prob_y']
    # res = pd.merge( res,df3,on=('orderid','geohashed_end_loc'),how='left' ).fillna(0.001)
    # res['prob'] = res['prob_x'] * res['prob_y']
    # del res['prob_x']
    # del res['prob_y']
    # res = pd.merge(res, df4, on=('orderid', 'geohashed_end_loc'),how='left').fillna(0.001)
    # res['prob'] = res['prob_x'] * res['prob_y']
    # del res['prob_x']
    # del res['prob_y']

    # groupby = res.sort_values('prob', ascending=False).drop_duplicates( ['orderid','geohashed_end_loc'] ).groupby('orderid', as_index=False)
    # n1 = groupby.nth(0)
    # n1.columns = ['orderid','pred1','prob1']
    # n2 = groupby.nth(1)
    # n2.columns = ['orderid','pred2','prob2']
    # n3 = groupby.nth(2)
    # n3.columns = ['orderid','pred3','prob3']
    # res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
    # res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
    # res = res[['orderid', 'pred1', 'pred2', 'pred3']]
    return res
tr = pd.read_csv('../data/train.csv')
df2 = trans_time(tr)
make_eval( df2 )

def make_sub():
    tr = pd.read_csv('../data/train.csv')
    df2 = trans_time(tr)
    # df2.to_csv('../data/data.csv', index=None)
    te = pd.read_csv('../data/test.csv')
    te = trans_time( te )

    user_hour_prob = get_user_hour_prob( df2,a=1.0 )
    user_start_prob = get_user_start_prob( df2,a=1.0 )
    user_weekday_prob = get_user_weekday_prob(df2, 1.0)
    user_quarter_prob = get_user_quarter_prob(df2, 1.0)

    # 这里假设时段和起点相互独立，则有全概率公式
    df1 = pd.merge(te[['orderid', 'userid', 'time_range']], user_hour_prob, on=('userid', 'time_range'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    df2 = pd.merge(te[['orderid', 'userid', 'geohashed_start_loc']], user_start_prob, on=('userid', 'geohashed_start_loc'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    df3 = pd.merge(te[['orderid', 'userid', 'weekday']], user_weekday_prob, on=('userid', 'weekday'))[
        ['orderid', 'geohashed_end_loc', 'prob']]
    df4 = pd.merge(te[['orderid', 'userid', 'quarter']], user_quarter_prob, on=('userid', 'quarter'))[
        ['orderid', 'geohashed_end_loc', 'prob']]

    res = pd.merge(df1, df2, on=('orderid', 'geohashed_end_loc'), how='left').fillna(0.001)
    res['prob'] = res['prob_x'] * res['prob_y']
    del res['prob_x']
    del res['prob_y']
    res = pd.merge(res, df3, on=('orderid', 'geohashed_end_loc'), how='left').fillna(0.001)
    res['prob'] = res['prob_x'] * res['prob_y']
    del res['prob_x']
    del res['prob_y']
    res = pd.merge(res, df4, on=('orderid', 'geohashed_end_loc'), how='left').fillna(0.001)
    res['prob'] = res['prob_x'] * res['prob_y']
    del res['prob_x']
    del res['prob_y']

    groupby = res.sort_values('prob', ascending=False).drop_duplicates(['orderid', 'geohashed_end_loc']).groupby(
        'orderid', as_index=False)
    n1 = groupby.nth(0)
    n1.columns = ['orderid', 'pred1', 'prob1']
    n2 = groupby.nth(1)
    n2.columns = ['orderid', 'pred2', 'prob2']
    n3 = groupby.nth(2)
    n3.columns = ['orderid', 'pred3', 'prob3']
    res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('wx4f9mu')
    res = pd.merge(res, n3, on=('orderid'), how='left').fillna('wx4e5sw')
    res = res[['orderid', 'pred1', 'pred2', 'pred3']]
    print(len(res)) #891517

    r = pd.read_csv('../res/syp_userid_loc.csv', names=['orderid', 'pred1', 'pred2', 'pred3'])
    a = pd.DataFrame()
    a['orderid'] = list(set(r['orderid'].unique()) - set(res['orderid'].unique()))

    r = pd.merge(r, a, on='orderid')
    res = res.append(r[['orderid', 'pred1', 'pred2', 'pred3']]).drop_duplicates('orderid')
    res.to_csv('../res/res_user.csv', header=None, index=None) #0.2407
# make_sub()

# if __name__ == "__main__":
#     d = pd.read_csv('data/data.csv')
#     c = make_eval(d)
#     b = pd.merge( d,c,on='orderid' )[['orderid','userid','geohashed_end_loc','pred1','pred2','pred3']]
#     bad_case = b[(b.geohashed_end_loc != b.pred1) & (b.geohashed_end_loc != b.pred2) & (b.geohashed_end_loc != b.pred3)]
#     dd = pd.merge(bad_case, d, on='orderid')
#     print b






