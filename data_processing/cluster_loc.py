import os
import cPickle
import pandas as pd
import numpy
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

from data_processing.data_utils import get_lat, get_lon

print "Generating location point list"
tr = pd.read_csv('../data/train.csv')
te = pd.read_csv('../data/test.csv')

loc_raw = set(tr['geohashed_start_loc'].values).union(
    set(te['geohashed_start_loc'].values).union(set(tr['geohashed_end_loc'].values)))

dests = []
for v in loc_raw:
    lat = get_lat(v)
    lon = get_lon(v)
    dests.append([lat, lon])
pts = numpy.array(dests)
print pts[:5]

with open(os.path.join('../data/', "arrivals.pkl"), "w") as f:
    cPickle.dump(pts, f, protocol=cPickle.HIGHEST_PROTOCOL)

print "Doing clustering"
# bw = estimate_bandwidth(pts, quantile=.005, n_samples=1000,random_state=20170630)
# print bw
bw = 0.01

ms = MeanShift( bandwidth=bw,bin_seeding=True,min_bin_freq=5)
ms.fit(pts)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(cluster_centers)
print "Clusters shape: ", cluster_centers.shape

###############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(pts[my_members, 0], pts[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

with open(os.path.join('../data/', "arrival-clusters.pkl"), "w") as f:
    cPickle.dump(cluster_centers, f, protocol=cPickle.HIGHEST_PROTOCOL)