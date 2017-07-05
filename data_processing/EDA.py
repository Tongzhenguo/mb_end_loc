# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
tr = pd.read_csv('../data/train.csv')

print(len(tr))  # 3214096
print(len(tr['userid'].unique()))  # 349693
print(len(tr['bikeid'].unique()))  # 485465
print(tr['biketype'].unique())  # [1,2]
print('time uniq',tr['starttime'].unique().shape[0] )

df = pd.to_datetime( tr['starttime'] )
print(df.min())  # 2017-05-10 00:00:09
print(df.max())  # Timestamp('2017-05-24 19:00:16')

print(len(list(tr['geohashed_start_loc'].unique()) + list(tr['geohashed_end_loc'].unique())))  # 178934
start_end_pairs =  tr[['geohashed_start_loc','geohashed_end_loc','orderid']].groupby(
     ['geohashed_start_loc','geohashed_end_loc'],as_index=False
 ).count().sort_values(['orderid'],ascending=False)
start_end_pairs.columns = ['geohashed_start_loc','geohashed_end_loc','times'] ##1420276
start_end_pairs[start_end_pairs.times>10]['times'].hist() # 频繁项长尾分布：大部分在行程路线只出现不到78次，最频繁行程有681次
plt.show()
print(start_end_pairs.head(10))
'''
       geohashed_start_loc geohashed_end_loc  times
696727             wx4f9ky           wx4f9mk    681
697517             wx4f9mk           wx4f9ky    497
696393             wx4f9kn           wx4f9mk    437
696395             wx4f9kn           wx4f9ms    372
867271             wx4fg87           wx4ferq    356
696729             wx4f9ky           wx4f9ms    356
708646             wx4f9wb           wx4f9mu    355
697694             wx4f9ms           wx4f9kn    345
416470             wx4eq0c           wx4eq23    323
697507             wx4f9mk           wx4f9kn    319
'''