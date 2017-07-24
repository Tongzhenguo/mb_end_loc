# coding=utf-8
import os

import math

import gc
import pandas as pd
import itertools

import pickle

"""
    基于推荐的算法：
    如果用户在训练集出现过，预测的概率公式是：
    prob(u,new_start) = prob(u,old_start)*sim(new_start,old_start|u)
    其中相似度算法一种方法是--共现矩阵算法
"""
def get_user_start_prob( df2,a ):
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
    return df

def get_rating_matrix( train ):
    path = '../cache/rating_all_test.pkl'
    if os.path.exists(path):
        train = pickle.load(open(path, "rb"))
    else:
        train = train.groupby(['userid', 'geohashed_end_loc', 'geohashed_start_loc'])['orderid'].count().reset_index()
        pickle.dump(train, open(path, 'wb'), True)  # dump 时如果指定了 protocol 为 True，压缩过后的文件的大小只有原来的文件的 30%
    return train

def get_concur_mat( train ):
    path = "../cache/concur_mat_test.pkl"
    if os.path.exists(path):
        sim_mat = pickle.load(open(path, "rb"))
    else:
        rat_mat = get_rating_matrix( train )
        sim_mat = pd.DataFrame()
        item1_list = []
        item2_list = []
        concur_count = []
        user_groups = rat_mat.groupby( ['userid','geohashed_end_loc'] )
        for name,group in user_groups:
            for pair in itertools.combinations(list(group['geohashed_start_loc'].values), 2):
                item1_list.append( pair[0] )
                item2_list.append( pair[1] )
                concur_count.append( 1 )
            # print name
        sim_mat['item1'] = item1_list
        sim_mat['item2'] = item2_list
        sim_mat['count'] = concur_count
        sim_mat = sim_mat.groupby(['item1', 'item2'], as_index=False).sum()
        pickle.dump(sim_mat, open(path, 'wb'), True)
    return sim_mat

def get_concur_sim( train ):
    path = "cache/concur_sim_mat_test.pkl"
    if os.path.exists(path):
        concur_mat = pickle.load(open(path, "rb"))
    else:
        concur_mat = get_concur_mat( train )
        rat_mat = get_rating_matrix( train )
        item_vector = rat_mat.groupby(['geohashed_start_loc'],as_index=False).count()
        item_vector.index = item_vector['geohashed_start_loc']
        item_vector.columns = ['userid', 'geohashed_end_loc', 'geohashed_start_loc','count']
        item_count_dict = item_vector['count'].to_dict()
        concur_mat['item1_count'] = concur_mat['item1'].apply( lambda p:item_count_dict[p] )
        concur_mat['item2_count'] = concur_mat['item2'].apply(lambda p: item_count_dict[p])
        concur_mat['sim'] = concur_mat['count'] / (concur_mat['item1_count'].apply(math.sqrt) * concur_mat['item2_count'].apply(math.sqrt))
        pickle.dump(concur_mat, open(path, 'wb'), True)
    return concur_mat[['item1','item2','sim']]

def get_sim_start( te ):
    #找到一个用户测试集新地点对应的最相似/最近的订单起点
    #prob(u,new_start) = prob(u,old_start)*sim(new_start,old_start)
    tr = pd.read_csv('../data/train.csv')[['geohashed_start_loc','userid']].drop_duplicates(['geohashed_start_loc','userid'])
    te = pd.read_csv('../data/test.csv')[['orderid','userid','geohashed_start_loc']]
    te['label'] = 1
    sim_mat = get_concur_sim( tr )
    sim_mat1 = sim_mat
    sim_mat1['t'] = sim_mat1['item1']
    sim_mat1['item1'] = sim_mat1['item2']
    sim_mat1['item2'] = sim_mat1['t']
    del sim_mat['t']
    sim_mat = sim_mat.append( sim_mat1 )
    del sim_mat1
    gc.collect()
    pd.merge( tr, )
    sim_mat = sim_mat.sort_values('sim',ascending=False).drop_duplicates('item1')
    df = pd.merge( tr,te,on=('geohashed_start_loc','userid'),how='left' ).fillna(0)
    df = pd.merge( df[df.label == 0],sim_mat,right_on='item1',left_on='geohashed_start_loc' )
    del df['label']
    del df['item1']

    # ss = pd.merge( tr,sim_mat,left_on='geohashed_start_loc',right_on='item1' )
    # del ss['item1']
    # gc.collect()
    #
    # ss = ss.sort_values(['userid','geohashed_start_loc','sim'],ascending=False).drop_duplicates(['userid','geohashed_start_loc'])
    # df = pd.merge(tr[['userid','geohashed_start_loc']].drop_duplicates(['userid','geohashed_start_loc']), te, on=('userid'))
    # df = pd.merge(sim_mat, te, left_on='item1',right_on='geohashed_start_loc',how='right' ).fillna(0)
    # df = df[df.label == 0].sort_values(['orderid','sim'],ascending=False).drop_duplicates('orderid')



# def make_eval(val):
#     user_start_prob = get_user_start_prob( val,1.0 )
#
#
#     # res = pd.DataFrame()
#     df2 = pd.merge(val[['orderid', 'userid', 'geohashed_start_loc']], user_start_prob, on=('userid', 'geohashed_start_loc'))[['orderid', 'geohashed_end_loc', 'prob']]
#     df3 = pd.merge(val[['orderid', 'userid', 'weekday']], user_weekday_prob, on=('userid', 'weekday'))[
#         ['orderid', 'geohashed_end_loc', 'prob']]
#     df4 = pd.merge(val[['orderid', 'userid', 'quarter']], user_quarter_prob, on=('userid', 'quarter'))[
#         ['orderid', 'geohashed_end_loc', 'prob']]
#
#     res = pd.merge( df1,df2,on=('orderid','geohashed_end_loc'),how='left' ).fillna(0.001) #保证p(end|hour)*p(end|start)大于只在end|start中出现的概率，忽略了他们的差异
#     res['prob'] = res['prob_x']*res['prob_y']
#     del res['prob_x']
#     del res['prob_y']
#     res = pd.merge( res,df3,on=('orderid','geohashed_end_loc'),how='left' ).fillna(0.001)
#     res['prob'] = res['prob_x'] * res['prob_y']
#     del res['prob_x']
#     del res['prob_y']
#     res = pd.merge(res, df4, on=('orderid', 'geohashed_end_loc'),how='left').fillna(0.001)
#     res['prob'] = res['prob_x'] * res['prob_y']
#     del res['prob_x']
#     del res['prob_y']
#
#     groupby = res.sort_values('prob', ascending=False).drop_duplicates( ['orderid','geohashed_end_loc'] ).groupby('orderid', as_index=False)
#     n1 = groupby.nth(0)
#     n1.columns = ['orderid','pred1','prob1']
#     n2 = groupby.nth(1)
#     n2.columns = ['orderid','pred2','prob2']
#     n3 = groupby.nth(2)
#     n3.columns = ['orderid','pred3','prob3']
#     res = pd.merge(n1, n2, on=('orderid'), how='left').fillna('-1')
#     res = pd.merge(res, n3, on=('orderid'), how='left').fillna('-1')
#     # res = res[['orderid', 'pred1', 'pred2', 'pred3']]
#     return res

def make_sub():
    #prob(u,new_start) = prob(u,old_start)*sim(new_start,old_start)
    tr = pd.read_csv('../data/train.csv')[['userid','geohashed_start_loc']].drop_duplicates(['userid','geohashed_start_loc'])
    te = pd.read_csv('../data/test.csv')[['userid','geohashed_start_loc']]
    te['label'] = 1
    sim_mat = get_concur_sim( tr )
    df = pd.merge( tr,te,on=('userid'),how='right' ).fillna(0)
    pd.merge( df[df.label==0],sim_mat,left_on='item1',right_on='geohashed_start_loc' )[['orderid','item2','sim']]
    pd.merge(df[df.label == 0], sim_mat, left_on='item2', right_on='geohashed_start_loc')[['orderid', 'item1', 'sim']]
# make_sub()

# if __name__ == "__main__":
#     d = pd.read_csv('data/data.csv')
#     c = make_eval(d)
#     b = pd.merge( d,c,on='orderid' )[['orderid','userid','geohashed_end_loc','pred1','pred2','pred3']]
#     bad_case = b[(b.geohashed_end_loc != b.pred1) & (b.geohashed_end_loc != b.pred2) & (b.geohashed_end_loc != b.pred3)]
#     dd = pd.merge(bad_case, d, on='orderid')
#     print b






