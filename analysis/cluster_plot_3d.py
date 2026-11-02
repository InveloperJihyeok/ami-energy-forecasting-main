import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from umap import UMAP

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from model.setup import IndexSetup, ColumnsSetup

if __name__ == "__main__":
    SEASON = sys.argv[1]
    REDUCE_METHOD = sys.argv[2]
    N_CLUSTER = int(sys.argv[3])

CONF = {'season':SEASON, 'prev_times':192, 'next_times':24, 
            'test_ratio':0.2, 'train_sample_ratio':1, 'summer_months':[4, 5, 6, 7, 8, 9],
            'time_flag_var':'time', 'time_var_format':'.+_d[1-7]_t_$', 'value_var_format':'^h[0-9]+'}

data = pd.read_pickle(f'{dir_path}/data/ireland_power_1h.bin')

INDEX_SETUP = IndexSetup(CONF, data)
COLUMNS_SETUP = ColumnsSetup(CONF, data) 
    
arr_v = data.loc[INDEX_SETUP.train_period_index, COLUMNS_SETUP.value_var_names].values.T
arr_v = np.log(arr_v + 1)

if REDUCE_METHOD == 'tsne':
    model_reduce = TSNE(n_components=3, perplexity=10)
elif REDUCE_METHOD == 'umap':
    model_reduce = UMAP(n_components=3)
else:
    model_reduce = ''
    
if model_reduce != '':
    arr_v = model_reduce.fit_transform(arr_v) 

model_clust = KMeans(n_clusters=N_CLUSTER, random_state=0)
clust_id_set = model_clust.fit_predict(arr_v)

df_cluster = pd.DataFrame(arr_v)
df_cluster['cluster'] = clust_id_set

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(N_CLUSTER):
    ax.scatter(df_cluster[df_cluster.cluster==i][0],df_cluster[df_cluster.cluster==i][1],df_cluster[df_cluster.cluster==i][2], s=5)

plt.show()