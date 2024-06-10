import numpy as np
import pandas as pd
from typing import Optional, Tuple
# torch
import torch
from torch import Tensor

class SplitEngine:

    def __init__(self,
                 df_data : pd.DataFrame,
                 df_pre_labels : pd.DataFrame,
                 doc_embs : Tensor,
                 label_embs : Tensor,
                 threshold : Optional[float]=0.65, # similarity threshold
                 label_red : Optional[bool]=False, # eliminate some predefined labels
                 min_samples : Optional[int]=5, # force assignment to pre-define labels
                 ) -> None:
        # we take a deep copy to not impact the org ata
        self.df_data = df_data.copy()
        self.df_data.reset_index(inplace=True) 
        self.df_label_info = pd.DataFrame(df_pre_labels.label_id.values, columns=['label_id'])
        self.doc_embs = doc_embs
        self.label_embs = label_embs
        self.threshold = threshold
        self.label_red = label_red
        self.min_samples = min_samples
        # to be computed
        self.sim_matrix = None
        self.df_assigned = None
        self.df_not_assigned = None
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Step 1: compute sim matrix
        # sim_matrix[i][j]: the similairty of i-th example against label j
        self.sim_matrix = self.sim_calc()
        sim_matrix_topk = self.sim_matrix.topk(1, dim=1)
        self.df_data['assign_sim'] = sim_matrix_topk.values.numpy().flatten()
        self.df_data['assign_label_id'] = sim_matrix_topk.indices.numpy().flatten() + 1
        self.df_data['assigned'] = self.df_data.assign_sim >= self.threshold

        # Step 2: initial assignment based on similarity vs threshold
        # assigned
        df_ass_tmp = self.df_data[self.df_data.assigned]
        df_assigned = df_ass_tmp.groupby('assign_label_id')[['index', 'text']].agg(list)
        df_assigned.reset_index(inplace=True)
        df_assigned['count'] = df_assigned['index'].apply(lambda x: len(x))
        # unassigned
        df_not_assigned = self.df_data[~self.df_data.assigned]

        # Step 3: we folk here
        # 1) pre-defined label reduction: some of pre-defined topics are not proper, we drop them
        # 2) ensure each pre-defined labels have x amount of labels
        if self.label_red:
            # finalize assigned
            self.df_assigned = df_assigned[['assign_label_id', 'index', 'text']].\
                rename(columns={'assign_label_id' : 'label_id'}).reset_index(drop=True)
            self.df_assigned['label_id'] = pd.factorize(self.df_assigned['label_id'])[0] + 1
            # unassigned 
            self.df_not_assigned = df_not_assigned[['index', 'text']].reset_index(drop=True)
        else:
            self.force_assignment(df_assigned, df_not_assigned)

        return self.df_assigned, self.df_not_assigned

    # potentially, you can define different ways of measuring similarities
    def sim_calc(self):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if len(self.doc_embs.shape) == 1:
            self.doc_embs = self.doc_embs.unsqueeze(0)
        if len(self.label_embs.shape) == 1:
            self.label_embs = self.label_embs.unsqueeze(0)
        a_norm = torch.nn.functional.normalize(self.doc_embs, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(self.label_embs, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def force_assignment(self, df_assigned : pd.DataFrame, df_not_assigned : pd.DataFrame):
        df_assigned = df_assigned.merge(
            self.df_label_info, how='right', left_on='assign_label_id', right_on=['label_id'])
        df_assigned['count'].fillna(0, inplace=True)
        df_assigned = df_assigned[['label_id', 'index', 'text', 'count']]
        # enhnace those labels that do not satisfy requirement
        df_tmp = df_assigned[df_assigned['count'] <= self.min_samples]
        container = []
        to_remove_from_unassigned = []
        for _, row in df_tmp.iterrows():
            label_id = row.label_id
            num_samples_needed = int(self.min_samples - row['count'])
            v = df_not_assigned[df_not_assigned.assign_label_id == label_id].sort_values(by=['assign_sim'], ascending=False)
            assert(len(v) >= num_samples_needed)
            v = v[:num_samples_needed]    
            cur_idx_set = v['index'].tolist() + (row['index'] if isinstance(row['index'], list) else [])
            cur_text_set = v['text'].tolist() + (row['text'] if isinstance(row['text'], list) else [])
            to_remove_from_unassigned += v['index'].tolist()
            container.append([label_id, cur_idx_set, cur_text_set, len(cur_text_set)])    
        self.df_assigned = df_assigned.append(pd.DataFrame(container, columns=df_assigned.columns))
        self.df_assigned = self.df_assigned.sort_values(by='count', ascending=False)
        self.df_assigned.drop_duplicates(subset='label_id', keep="first", inplace=True)
        self.df_assigned.sort_values(by='label_id', inplace=True)
        # remove them from unassigned
        self.df_not_assigned = df_not_assigned[~df_not_assigned['index'].isin(to_remove_from_unassigned)]
        self.df_not_assigned = self.df_not_assigned[['index', 'text']].reset_index(drop=True)

    @property
    def assigned_doc(self):
        return self.df_assigned
    
    @property
    def not_assgined_doc(self):
        return self.df_not_assigned
    
    @property
    def similairty_matrix(self):
        return self.sim_matrix