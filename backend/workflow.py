import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple
# torch
import torch
from torch import Tensor
import torch.nn.functional as F
# local
from .llm import BaseLLM
from .utils import doc_cos_sim
from .split_engine import SplitEngine
from .grouper_engine import Grouper
from .keyword_extractor import KeywordExtractor

### default configuration for auto-labeler
AUTO_LABLER_DEFAULT_CONFIG = {
    "device" : "cpu", # or cuda
    "llm" : "sentence_transformer",
    # docs emb
    "load_emb_if_exists" : True,
    # predefined labels
    "pred_sim_measure" : "cosine", # similarity measure for initial assignment
    "pred_sim_threshold" : 0.65, # threshold above which sample will be assigned
    "pred_min_sample" : 5, # if 0, elminate those topics has nothing assigned, if > 0, the number of samples are enforced to be assigned
    # genai...
    # clustering config
    "n_cluster" : 3,
    "precomputed" : False,
    "clustering_method" : "kmeans",
    "dim_reduction" : "pca",
    "red_dim" : 5, 
    # kw extractor
    "keep_top_k" : 5
}

AUTO_LABEL_RUN_CONFIG = {
    "workspace" : "AutoLabler",
    "encode_all_doc_with_ai" : { "ai" : True, "load_if_exists" : True},
    "preprocess_predefined_label" : { "run" : True, "load_if_exists" : True},
    "group_docs" : {"run" : True, "load_if_exists" : True}
}

class AutoLabeler:

    def __init__(self, 
                 input_df : pd.DataFrame, 
                 pred_labels : Optional[list]=None,
                 config : Optional[dict]=AUTO_LABLER_DEFAULT_CONFIG,
                 config_run : Optional[dict]=AUTO_LABEL_RUN_CONFIG) -> None:
        """
        Args:
            input_df (pd.DataFrame): input DataFrame with schema | text |
            pred_labels (list, optional): a set of pre-defined labels (if applicable). Default: None
            config (dict, optional): configuration for auto label. Default: AUTO_LABLER_DEFAULT_CONFIG
            coconfig_runnfig (dict, optional): configuration for auto label run. Default: AUTO_LABEL_RUN_CONFIG
        """
        self.df = input_df
        self.config = config
        self.config_run = config_run
        self.device = torch.device(self.config['device'])
        ## create workspace for auto-labler
        self.work_space = self.config_run['workspace']
        if not os.path.exists(self.work_space):
            os.makedirs(self.work_space)
        print(f"Auto-Labeler Workspace is created /'{self.work_space}/'.")
        ## load encoder
        self.encoder_model = BaseLLM.create_llm_model(self.config['llm'], self.device)
        ## predefined model
        self.df_pred_labels = None
        if pred_labels is not None:
            assert isinstance(pred_labels, list)
            self.df_pred_labels = pd.DataFrame(pred_labels, columns=['label'])
            self.df_pred_labels['label_id'] = np.arange(len(pred_labels)) + 1 # label starts from 1
        ## register grouper model
        self.grouper = Grouper(
            cluster_method=self.config['clustering_method'],
            n_clusters=self.config['n_cluster'],
            dim_red_method=self.config['dim_reduction'],
            reduced_dim=self.config['red_dim'])
        ## variables to be filled
        self.df_to_group = self.df.copy()
        self.df_groupped = None
        self.pred_labels_emb = None # the embedding of pre_defined labels (optional)
        self.doc_grouper = {'reducer' : None, 'clusterer' : None}
        self.topic_dict = dict() # for lda
        self.df_kw_summary = None
        
    ### encode all docs
    def encode_all_doc(self, use_summary : Optional[bool]=False):
        emb_file = os.path.join(self.work_space, "all_doc_summary_emb.pt" if use_summary else "all_doc_emb.pt")
        if os.path.exists(emb_file):
            return torch.load(emb_file)
        # embedding in-the-making
        doc_embs = self.encoder_model.encode(self.df_to_group.text.values.tolist())
        # save
        torch.save(doc_embs, emb_file)
        return doc_embs

    ### encoding:
    # 1) use_ai :    gen-AI first convert text => summarized text, then encode them with vanilla transformer
    # 2) w/o use_ai: encode with transformer
    def encode_all_doc_with_ai(self):
        if not self.config_run[self.encode_all_doc_with_ai.__name__]['ai']:
            # just encode the original text
            self.all_docs_emb = self.encode_all_doc()
            return
        file_name = os.path.join(self.work_space, "df_to_group_use_summary.csv")
        if self.config_run[self.encode_all_doc_with_ai.__name__]['load_if_exists'] and os.path.exists(file_name):
            # load from exist ?
            tmp = pd.read_csv(file_name)
            assert len(self.df_to_group) == len(tmp)
            self.df_to_group = tmp
            self.all_docs_emb = self.encode_all_doc(True)
            self.df['text_summary'] = self.df_to_group['text']
            return

        ## TODO: add GenAI encode/decode
        self.df_to_group['text'] = self.df_to_group['text'] # override
        self.df['text_summary'] = self.df_to_group['text'] # add summary column to org table
        self.df_to_group.to_csv(file_name, index=False)
        self.all_docs_emb = self.encode_all_doc(True)

    ### Split data into two groups: 1) fitted into pre-defined labels; 2) failed to be assigned
    def preprocess_predefined_label(self):
        file_grouped = os.path.join(self.work_space, "df_grouped.csv")
        file_to_group = os.path.join(self.work_space, "df_to_group.csv")
        if self.df_pred_labels is None:
            # no pre-defined label
            return
        if not self.config_run[self.preprocess_predefined_label.__name__]['run']:
            # run ?
            return
        if self.config_run[self.preprocess_predefined_label.__name__]['load_if_exists'] and \
            os.path.exists(file_grouped) and os.path.exists(file_to_group):
            # load from exist ?
            self.df_groupped = pd.read_csv(file_grouped)
            self.df_to_group = pd.read_csv(file_to_group)
            return
        
        ## 1) Encode pre-defined labels (very small size)
        self.pred_labels_emb = self.encoder_model.encode(self.df_pred_labels.label.values.tolist())

        ## 2) Split df to df_assigned, df_not_assigned
        split_engine = SplitEngine(self.df_to_group, self.all_docs_emb, self.df_pred_labels, self.pred_labels_emb,
                    self.config['pred_sim_measure'], self.config['pred_sim_threshold'], self.config['pred_min_sample'])
        split_engine.run()
        if split_engine.assigned_doc is None:
            # do nothing
            return
        self.df_groupped = split_engine.assigned_doc
        self.df_to_group = split_engine.not_assgined_doc
        self.df_groupped.to_csv(file_grouped, index=False)
        self.df_to_group.to_csv(file_to_group, index=False)

    ### run groupping agorithm
    def group_docs(self):
        file_name = os.path.join(self.work_space, "df_to_group_finished.csv")
        if self.config_run[self.group_docs.__name__]['run'] and os.path.exists(file_name):
            # load from exist ?
            self.df_to_group = pd.read_csv(file_name)
            if self.config['clustering_method']=='lda':
                Grouper.load_models(self.doc_grouper, self.work_space, True)
                f = open(os.path.join(self.work_space, "lda_topics.json"), 'r')
                self.topic_dict = json.loads(f)
            else:
                Grouper.load_models(self.doc_grouper, self.work_space)
            return

        dist_matrix = None
        doc_embs = self.all_docs_emb[self.df_to_group['id'].tolist()]
        if self.config['precomputed']:
            dist_matrix = doc_cos_sim(doc_embs, doc_embs)
            dist_matrix = dist_matrix.detach().numpy().astype('float64')
        doc_embs = doc_embs.detach().numpy().astype('float64')
        self.df_to_group, self.topic_dict = self.grouper.run(self.df_to_group, doc_embs, dist_matrix if self.config['precomputed'] else None)
        self.df_to_group.to_csv(file_name, index=False)
        self.grouper.export_models(self.work_space)
        if self.topic_dict is not None:
            f = open(os.path.join(self.work_space, 'lda_topics.json'), 'w')
            json.dump(self.topic_dict, f)

    ### run kw extractor
    def kw_extraction(self):
        kw_extractor = KeywordExtractor(self.df_groupped, self.df_to_group, self.config['keep_top_k'])
        kw_extractor.run()
        self.df_kw_summary = kw_extractor.get_topics_and_docs(self.df, 5)

    


        
