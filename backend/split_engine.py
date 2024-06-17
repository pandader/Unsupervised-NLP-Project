import pandas as pd
from typing import Optional
# torch
from torch import Tensor
# local
from .utils import doc_cos_sim

class SplitEngine:

    def __init__(self,
                 df_data : pd.DataFrame,
                 doc_embs : Tensor,
                 df_pre_labels : pd.DataFrame,
                 label_embs : Tensor,                 
                 sim_measure : Optional[str]="cosine",
                 threshold : Optional[float]=0.65,
                 min_samples : Optional[int]=5,
                 ) -> None:
        """
        Args:
            df_data (pd.DataFrame): input data
            doc_embs (Tensor): embeddings of all docs 
            df_pre_labels (pd.DataFrame): predefined labels and their ids, i.e., schema |label_text | label_id |
            label_embs (Tensor): embedding of all predefined labels
            sim_measure (str): similarity measure. Default: cosine
            threshold (float, optional): threshold above which sample will be assigned. Default: 0.65
            min_samples (int, optional): if 0, elminate those topics has nothing assigned, if > 0, the number of samples are enforced to be assigned. Default: 5
        """
        self.df_data = df_data
        self.doc_embs = doc_embs
        self.df_label_info = df_pre_labels 
        self.label_embs = label_embs
        self.sim_measure = sim_measure
        self.threshold = threshold
        self.min_samples = min_samples
        # to be computed
        self.sim_matrix = None
        self.df_assigned = None
        self.df_not_assigned = None
    
    ### execution
    def run(self):
        
        ## 1) make similarity statistics
        self.sim_matrix = self.sim_calc()
        sim_matrix_topk = self.sim_matrix.topk(1, dim=1)
        self.df_data['assign_sim'] = sim_matrix_topk.values.numpy().flatten()
        self.df_data['assign_label_id'] = sim_matrix_topk.indices.numpy().flatten() + 1
        self.df_data['assigned'] = self.df_data.assign_sim >= self.threshold
        if len(self.df_data[self.df_data.assigned]) == 0:
            print("Cannot assign any text to pre-defined labels with current threshold, resume standard groupping process.")
            return

        ## 2) initial assignment based on similarity vs threshold
        # assigned
        df_ass_tmp = self.df_data[self.df_data.assigned]
        df_assigned = df_ass_tmp.groupby('assign_label_id')[['id', 'text']].agg(list)
        df_assigned.reset_index(inplace=True)
        df_assigned['count'] = df_assigned['id'].apply(lambda x: len(x))
        # unassigned
        df_not_assigned = self.df_data[~self.df_data.assigned]

        ## 3: we folk here
        #    1) min_samples = 0: pre-defined label reduction, i.e. some of pre-defined topics are not proper, we drop them
        #    2) min_samples > 0: ensure each pre-defined labels have (x) amount of labels
        if self.min_samples > 0:
            self.force_assignment(df_assigned, df_not_assigned)    
        # finalize assigned
        self.df_assigned = df_assigned[['assign_label_id', 'id', 'text']].rename(columns={'assign_label_id' : 'label_id'}).reset_index(drop=True)
        self.df_assigned['label_id'] = pd.factorize(self.df_assigned['label_id'])[0] + 1
        # unassigned 
        self.df_not_assigned = df_not_assigned[['id', 'text']].reset_index(drop=True)
    
    # can customize 
    def sim_calc(self):
        # sim_matrix[i][j]: the similairty of i-th example against label j
        if self.sim_measure == "cosine":
            return doc_cos_sim(self.doc_embs, self.label_embs)
        else:
            raise Exception(f"Unsupported similarity measure {self.sim_measure}.")

    ### ensure minimum number of samples for each pre-defined class
    def force_assignment(self, df_assigned : pd.DataFrame, df_not_assigned : pd.DataFrame):
        df_assigned = df_assigned.merge(
            self.df_label_info, how='right', left_on='assign_label_id', right_on=['label_id'])
        df_assigned['count'].fillna(0, inplace=True)
        df_assigned = df_assigned[['label_id', 'id', 'text', 'count']]
        # enhnace those labels that do not satisfy requirement
        df_tmp = df_assigned[df_assigned['count'] <= self.min_samples]
        container = []
        to_remove_from_unassigned = []
        for _, row in df_tmp.iterrows():
            label_id = row.label_id
            num_samples_needed = int(self.min_samples - row['count'])
            v = df_not_assigned[df_not_assigned.assign_label_id == label_id].sort_values(by=['assign_sim'], ascending=False)
            if len(v) < num_samples_needed:
                continue
            v = v[:num_samples_needed]    
            cur_idx_set = v['id'].tolist() + (row['id'] if isinstance(row['id'], list) else [])
            cur_text_set = v['text'].tolist() + (row['text'] if isinstance(row['text'], list) else [])
            to_remove_from_unassigned += v['id'].tolist()
            container.append([label_id, cur_idx_set, cur_text_set, len(cur_text_set)])    
        self.df_assigned = df_assigned.append(pd.DataFrame(container, columns=df_assigned.columns))
        self.df_assigned = self.df_assigned.sort_values(by='count', ascending=False)
        self.df_assigned.drop_duplicates(subset='label_id', keep="first", inplace=True)
        self.df_assigned.sort_values(by='label_id', inplace=True)
        # remove them from unassigned
        self.df_not_assigned = df_not_assigned[~df_not_assigned['id'].isin(to_remove_from_unassigned)]
        self.df_not_assigned = self.df_not_assigned[['id', 'text']].reset_index(drop=True)

    @property
    def assigned_doc(self):
        return self.df_assigned
    
    @property
    def not_assgined_doc(self):
        return self.df_not_assigned
    
    @property
    def similairty_matrix(self):
        return self.sim_matrix