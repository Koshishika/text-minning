import pandas as pd
import codecs
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import codecs
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import _kmeans
from sklearn.cluster import KMeans

#pancake_df = pd.read_csv('dataset_tf_cluster.csv',encoding='ms932', sep=',')
csv_input = pd.read_csv('fortravel_bow.csv', encoding='ms932', sep=',',skiprows=0)
fortravel_df = csv_input.iloc[:,3:]

fortravel_val = fortravel_df.values
header = fortravel_df.columns.tolist()

def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
	#
    #return np.arccos(cosine_similarity(X, Y))/np.pi
	return cosine_similarity(X)

_kmeans.euclidean_distances = new_euclidean_distances 
k_means4 = _kmeans.KMeans(n_clusters =5,random_state = 0)
#_ = km.fit(pancake_val)
k_means4.fit(fortravel_val)

#k_means4= KMeans(n_clusters=4)
#pred=k_means4.fit_predict(pancake_val)
centers = k_means4.cluster_centers_

headers=fortravel_df.columns.tolist()

