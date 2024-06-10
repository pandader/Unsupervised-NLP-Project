import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from umap import UMAP
from umap import plot as umap_plot
# torch
from torch import Tensor

UMAPDefaultConfig = {
    'n_neighbors' : 15,
    'min_dist' : 0.2,
    'n_components' : 2,
    'metric' : 'euclidean'
}

class Groupper:

    def __init__(self, 
                 df_info : pd.DataFrame,
                 doc_embs : Tensor,
                 groupper_config : dict,
                 groupper_back_end : Optional[str]='kmean',
                 plotter_backend: Optional[str]='umap',
                 plotter_config : Optional[dict]=None
                 ) -> None:
        
        self.df_info = df_info
        self.x = doc_embs[df_info['index'].tolist()].detach().numpy()
        if groupper_back_end == 'kmean':
            self.grp_backend = KMeans(n_clusters=groupper_config['n_clusters'])
        else:
            raise Exception("Only support K-mean for now.")
        if plotter_backend == 'umap':
            if plotter_config is None:
                plotter_config = UMAPDefaultConfig
            self.plotter = UMAP(
                n_neighbors=UMAPDefaultConfig['n_neighbors'], 
                min_dist=UMAPDefaultConfig['min_dist'], 
                n_components=UMAPDefaultConfig['n_components'], 
                metric=UMAPDefaultConfig['metric'])
        else:
            raise Exception("Only support umap for now.")

    def run(self):
        fitted = self.grp_backend.fit(self.x)
        self.df_info['label_id'] = fitted.labels_ + 1
        # re-organize
        df_output = self.df_info.groupby('label_id')[['index', 'text']].agg(list)
        df_output.reset_index(inplace=True)
        df_output['count'] = df_output['index'].apply(lambda x: len(x))
        return df_output

    def plot(self):
        mapper = self.plotter.fit(self.x)
        umap_plot.points(mapper, labels=self.df_info['label_id'])



