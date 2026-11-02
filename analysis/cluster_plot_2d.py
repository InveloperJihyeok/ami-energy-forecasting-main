import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from umap import UMAP

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from model.setup import IndexSetup, ColumnsSetup

for seoson in ['summer', "winter"]:
    CONF = {'season':seoson, 'prev_times':192, 'next_times':24, 
                'test_ratio':0.2, 'train_sample_ratio':1, 'summer_months':[4, 5, 6, 7, 8, 9],
                'time_flag_var':'time', 'time_var_format':'.+_d[1-7]_t_$', 'value_var_format':'^h[0-9]+'}

    data = pd.read_pickle(f'{dir_path}/data/ireland_power_1h.bin')

    INDEX_SETUP = IndexSetup(CONF, data)
    COLUMNS_SETUP = ColumnsSetup(CONF, data) 
        
    arr_v = data.loc[INDEX_SETUP.train_period_index, COLUMNS_SETUP.value_var_names].values.T
    arr_v = np.log(arr_v + 1)

    arr_tsne = TSNE().fit_transform(arr_v)
    arr_umap = UMAP().fit_transform(arr_v)

    # K-Means
    for N_CLUSTER in range(2,3):
        model_clust = KMeans(n_clusters=N_CLUSTER, random_state=0)
        id_tsne = model_clust.fit_predict(arr_tsne)
        id_umap = model_clust.fit_predict(arr_umap)
        id_raw = model_clust.fit_predict(arr_v)

        print(f"\n[K-Means 실루엣 점수({seoson})]")
        silhouette_avg = silhouette_score(arr_tsne, id_tsne)
        print("t-SNE:", silhouette_avg)
        silhouette_avg = silhouette_score(arr_umap, id_umap)
        print("UMAP:", silhouette_avg)
        silhouette_avg = silhouette_score(arr_v, id_raw)
        print("Raw:", silhouette_avg)
        
        df_tsne = pd.DataFrame(arr_tsne)
        df_tsne['cluster'] = id_tsne
        for i in range(N_CLUSTER):
            plt.scatter(df_tsne[df_tsne.cluster==i][0],df_tsne[df_tsne.cluster==i][1], s=5)
        title = f"nc{N_CLUSTER}_tsne_{seoson}"
        plt.title(title)
        plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
        plt.clf()

        df_umap = pd.DataFrame(arr_umap)
        df_umap['cluster'] = id_umap
        for i in range(4):
            plt.scatter(df_umap[df_umap.cluster==i][0],df_umap[df_umap.cluster==i][1], s=5)
        title = f"nc{N_CLUSTER}_umap_{seoson}"
        plt.title(title)
        plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
        plt.clf()
        
        df_raw = pd.DataFrame(arr_umap)
        df_raw['cluster'] = id_raw
        for i in range(N_CLUSTER):
            plt.scatter(df_raw[df_raw.cluster==i][0],df_raw[df_raw.cluster==i][1], s=5)
        title = f"nc{N_CLUSTER}_raw_{seoson}"
        plt.title(title)
        plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
        plt.clf()
        
    # DBSCAN
    print(f"DBSCAN {seoson}")
    id_raw = DBSCAN(eps=30).fit_predict(arr_v)
    id_tsne = DBSCAN(eps=1).fit_predict(arr_tsne)
    id_umap = DBSCAN(eps=0.25).fit_predict(arr_umap)

    print(f"\n[DBSCAN 실루엣 점수({seoson})]")
    silhouette_avg = silhouette_score(arr_tsne, id_tsne)
    print("t-SNE:", silhouette_avg)
    silhouette_avg = silhouette_score(arr_umap, id_umap)
    print("UMAP:", silhouette_avg)
    silhouette_avg = silhouette_score(arr_v, id_raw)
    print("Raw:", silhouette_avg)

    print("Raw:",len(id_raw))
    for i in set(id_raw):
        print(f"{i}: {len(id_raw[id_raw==i])}")
    print("t-SNE:",len(id_tsne))
    for i in set(id_tsne):
        print(f"{i}: {len(id_tsne[id_tsne==i])}")
    print("UMAP:",len(id_umap))
    for i in set(id_umap):
        print(f"{i}: {len(id_umap[id_umap==i])}")

    df_tsne = pd.DataFrame(arr_tsne)
    df_tsne['cluster'] = id_tsne
    for i in range(max(id_tsne)+1):
        plt.scatter(df_tsne[df_tsne.cluster==i][0],df_tsne[df_tsne.cluster==i][1], s=5)
    title = f"dbscan_tsne_{seoson}"
    plt.title(title)
    plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    plt.clf()

    df_umap = pd.DataFrame(arr_umap)
    df_umap['cluster'] = id_umap
    for i in range(max(id_umap)+1):
        plt.scatter(df_umap[df_umap.cluster==i][0],df_umap[df_umap.cluster==i][1], s=5)
    title = f"dbscan_umap_{seoson}"
    plt.title(title)
    plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    plt.clf()
    
    df_raw = pd.DataFrame(arr_umap)
    df_raw['cluster'] = id_raw
    for i in range(max(id_raw)+1):
        plt.scatter(df_raw[df_raw.cluster==i][0],df_raw[df_raw.cluster==i][1], s=5)
    title = f"dbscan_raw_{seoson}"
    plt.title(title)
    plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    plt.clf()
    
    # # GMM
    # print(f"GMM {seoson}")
    # model_clust = GaussianMixture(n_components=2)
    # id_tsne = model_clust.fit_predict(arr_tsne)
    # id_umap = model_clust.fit_predict(arr_umap)
    # id_raw = model_clust.fit_predict(arr_v)
    
    # df_tsne = pd.DataFrame(arr_tsne)
    # df_tsne['cluster'] = id_tsne
    # for i in range(max(id_tsne)+1):
    #     plt.scatter(df_tsne[df_tsne.cluster==i][0],df_tsne[df_tsne.cluster==i][1], s=5)
    # title = f"gmm_tsne_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()

    # df_umap = pd.DataFrame(arr_umap)
    # df_umap['cluster'] = id_umap
    # for i in range(max(id_umap)+1):
    #     plt.scatter(df_umap[df_umap.cluster==i][0],df_umap[df_umap.cluster==i][1], s=5)
    # title = f"gmm_umap_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()
    
    # df_raw = pd.DataFrame(arr_umap)
    # df_raw['cluster'] = id_raw
    # for i in range(max(id_raw)+1):
    #     plt.scatter(df_raw[df_raw.cluster==i][0],df_raw[df_raw.cluster==i][1], s=5)
    # title = f"gmm_raw_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()
    
    # # Mean Shift
    # print(f"Mean Shift {seoson}")
    # model_clust = MeanShift()
    # id_tsne = model_clust.fit_predict(arr_tsne)
    # id_umap = model_clust.fit_predict(arr_umap)
    # id_raw = model_clust.fit_predict(arr_v)
    
    # df_tsne = pd.DataFrame(arr_tsne)
    # df_tsne['cluster'] = id_tsne
    # for i in range(max(id_tsne)+1):
    #     plt.scatter(df_tsne[df_tsne.cluster==i][0],df_tsne[df_tsne.cluster==i][1], s=5)
    # title = f"meanshift_tsne_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()

    # df_umap = pd.DataFrame(arr_umap)
    # df_umap['cluster'] = id_umap
    # for i in range(max(id_umap)+1):
    #     plt.scatter(df_umap[df_umap.cluster==i][0],df_umap[df_umap.cluster==i][1], s=5)
    # title = f"meanshift_umap_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()
    
    # df_raw = pd.DataFrame(arr_umap)
    # df_raw['cluster'] = id_raw
    # for i in range(max(id_raw)+1):
    #     plt.scatter(df_raw[df_raw.cluster==i][0],df_raw[df_raw.cluster==i][1], s=5)
    # title = f"meanshift_raw_{seoson}"
    # plt.title(title)
    # plt.savefig(f"{dir_path}/analyzer/figures/cluster/{title}.png")
    # plt.clf()