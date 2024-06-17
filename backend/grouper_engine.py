import os
import joblib
import numpy as np
import pandas as pd
import pickle as pk
from typing import Optional
# sklearn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# local
from .utils import tokenize_and_remove_stop_words

### UMAP Config
UMAP_CONFIG = {
    'n_neighbors' : 30, # controls how UMAP balances local versus global structure in the data (recomend 20~30)
    'min_dist' : 0.1, # provides the minimum distance apart that points are allowed to be in the low dimensional representation. (recommend ~0.1, larger values prevent packing points together)
}

### HDBSCAN Config
HDBSCAN_CONFIG = {
    'metric' : 'euclidean', # euclidean / precomputed
    'min_cluster_size' : 5, # the smallest size grouping that you wish to consider a cluster (Default: 5)
    'min_samples' : None # the larger the value of min_samples you provide, the more conservative the clustering (Default: None)
}

### Embedding space groupper
class Grouper:

    def __init__(self, 
                 cluster_method : str,
                 n_clusters : Optional[int]=-1,
                 dim_red_method : Optional[str]="",
                 reduced_dim : Optional[int]=5,
                 random_state : Optional[int]=42,
                 hdbscan_config : Optional[dict]=HDBSCAN_CONFIG,
                 umap_config : Optional[dict]=UMAP_CONFIG) -> None:
        
        ### set up dimension reducer
        self.dim_reducer = None
        if dim_red_method != "":
            if dim_red_method == "pca":
                self.dim_reducer = PCA(n_components=reduced_dim)
            elif dim_red_method == "umap":
                import umap
                self.dim_reducer = umap.UMAP(
                    n_components=reduced_dim,
                    n_neighbors=umap_config['n_neighbors'], 
                    min_dist=umap_config['min_dist'],
                    random_state=random_state,
                    n_jobs=-1)
            else:
                raise Exception(f"{dim_red_method} is not valid dimension reducer.")
        ### set up clusterer
        self.clusterer = None
        self.cluster_method = cluster_method
        if self.cluster_method == "kmeans":
            assert n_clusters > 1
            self.clusterer = KMeans(n_clusters=n_clusters)
        elif self.cluster_method == "hdbscan":
            import hdbscan
            self.clusterer = hdbscan.HDBSCAN(
                metric=hdbscan_config['metric'],
                min_cluster_size=hdbscan_config['min_cluster_size'],
                min_samples=hdbscan_config['min_samples']
            )
        elif self.cluster_method == "lda":
            self.n_clusters = n_clusters
        else:
            raise Exception(f"{self.cluster_method} is not valid dimension reducer.")

    def run(self, 
            input_df : pd.DataFrame, 
            doc_embs : np.ndarray, 
            pair_wise_dist : Optional[np.ndarray]=None):
        
        if self.cluster_method == 'lda':
            import gensim
            import gensim.corpora as corpora

            v = input_df.text.apply(lambda x: tokenize_and_remove_stop_words(x))
            data_words = v.values.tolist()
            id2word = corpora.Dictionary(data_words)
            corpus = [id2word.doc2bow(text) for text in data_words]
            # build lda model
            self.clusterer = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=self.n_clusters)
            # get topics
            topic_dict = {each[0] + 1 : each[1] for each in self.clusterer.print_topics()}
            label_ids = []
            for each in self.clusterer[corpus]:
                each.sort(key=lambda x: x[1])
                label_ids.append(each[-1][0])
            input_df['label_id'] = label_ids
            input_df = input_df.groupby('label_id')[['id', 'text']].agg(list)
            input_df.reset_index(inplace=True)
            input_df['count'] = input_df['id'].apply(lambda x: len(x))
            return input_df, topic_dict

        ## 1) dimension reduction (optional)
        x = doc_embs
        if self.dim_reducer is not None:
            x = self.dim_reducer.fit_transform(doc_embs)
            x = x.astype('float64')
        
        ## 2) clustering
        self.clusterer.fit(pair_wise_dist if pair_wise_dist is not None else x)
        if self.cluster_method == 'hdbscan':            
            input_df['label_id'] = self.clusterer.labels_ + 1
            input_df['prob'] = self.clusterer.probabilities_
            input_df = input_df[input_df.label_id != 0]
        else:
            self.clusterer.fit(x)
            input_df['label_id'] =  self.clusterer.predict(x) + 1
        input_df = input_df.groupby('label_id')[['id', 'text']].agg(list)
        input_df.reset_index(inplace=True)
        
        return input_df, None

    def export_models(self, path: str):
        if self.cluster_method == "lda":
            self.clusterer.save(os.path.join(path, "clusterer.model"))
            return
        # save dim reducer (if exists)
        if self.dim_reducer is not None:
            joblib.dump(self.dim_reducer, os.path.join(path, "dim_reducer.sav"))
        # save clusterer        
        joblib.dump(self.clusterer, os.path.join(path, "clusterer.sav"))

    @staticmethod
    def load_models(model_container : dict, path : str, is_lda : Optional[bool]=False):
        if is_lda:
            import gensim
            # later on, load trained model from file
            model_container['clusterer'] =  gensim.models.LdaModel.load(os.path.join(path, "clusterer.model"))  
            return

        if os.path.exists(os.path.join(path, "dim_reducer.sav")):
            model_container['reducer'] = joblib.load(os.path.join(path, "dim_reducer.sav"))
        model_container['clusterer'] = joblib.load(os.path.join(path, "clusterer.sav"))

    

    






