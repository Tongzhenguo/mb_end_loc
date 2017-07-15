# coding=utf-8
import os
import cPickle
import pandas as pd
import numpy
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from data_processing.data_utils import get_lat,get_lon
if __name__ == '__main__':
    print "Generating location point list"
    tr = pd.read_pickle('../data/tr_feat.pkl')
    te = pd.read_pickle('../data/te_feat.pkl')

    loc_raw = set(tr['geohashed_start_loc'].values).union(
        set(te['geohashed_start_loc'].values).union(set(tr['geohashed_end_loc'].values)))

    dests = []
    for v in loc_raw:
        lat,lon = get_lat(v ),get_lon(v )
        dests.append([lat, lon])
    pts = numpy.array(dests)
    print pts[:5]
    print "Doing clustering"
    # bw = estimate_bandwidth(pts, quantile=.005, n_samples=1000,random_state=20170630)
    # print bw
    bw = 0.005

    ms = MeanShift( bandwidth=bw,bin_seeding=True,min_bin_freq=3,n_jobs=4)
    ms.fit(pts)
    with open(os.path.join('../data/', "mean-shift-model.pkl"), "w") as f:
        cPickle.dump(ms, f, protocol=cPickle.HIGHEST_PROTOCOL)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = len(cluster_centers)
    print "Clusters shape: ", cluster_centers.shape #(2857L, 2L)

    # ###############################################################################
    # # Plot result
    # plt.figure(1)
    # plt.clf()
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     plt.plot(pts[my_members, 0], pts[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()


    end_loc = tr[['geohashed_end_loc']]
    end_loc['lat'] = end_loc['geohashed_end_loc'].apply(get_lat)
    end_loc['lon'] = end_loc['geohashed_end_loc'].apply(get_lon)

    end_loc['label'] = ms.predict(end_loc[['lat', 'lon']].values)
    print(end_loc[['label']].head())
    pd.to_pickle( end_loc[['label']],'../data/label.pkl' )


    ##这里计算一个DF,一列所属的label,另外3列是出现次数最多的终点候选，没有终点候选测试集的起点来补齐，默认是0
    end_loc['times'] = 1
    end_loc = end_loc[['geohashed_end_loc','label','times']].groupby(['geohashed_end_loc','label'],as_index=False).sum()
    end_loc.columns = ['loc','label','times']

    start_loc = te[['geohashed_start_loc']].drop_duplicates()
    start_loc['times'] = 0
    start_loc['lat'] = start_loc['geohashed_start_loc'].apply(get_lat)
    start_loc['lon'] = start_loc['geohashed_start_loc'].apply(get_lon)
    start_loc['label'] = ms.predict(start_loc[['lat', 'lon']].values)
    start_loc = start_loc[['geohashed_start_loc', 'label','times']].groupby(['geohashed_start_loc', 'label'], as_index=False).sum()
    start_loc.columns = ['loc', 'label', 'times']

    end_loc = end_loc.append( start_loc )
    print( end_loc.head() )

    rn1 = end_loc.sort_values('times', ascending=False).groupby(['label'],as_index=False).nth(0)
    rn1.columns = ['pred1','label','times']
    rn2 = end_loc.sort_values('times', ascending=False).groupby(['label'],as_index=False).nth(1)
    rn2.columns = ['pred2', 'label', 'times']
    rn3 = end_loc.sort_values('times', ascending=False).groupby(['label'],as_index=False).nth(2)
    rn3.columns = ['pred3', 'label', 'times']

    res = pd.merge(rn1, rn2, on=('label'), how='left').fillna('-1')
    res = pd.merge(res, rn3, on=('label'), how='left').fillna('-1')
    res = res[['label', 'pred1', 'pred2', 'pred3']]

    res.to_csv('../data/cluter_3_hot.csv',index=False)