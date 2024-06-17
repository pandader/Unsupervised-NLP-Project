from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class LLM_TYPE(Enum):
    SENTENCE_TRANSFORMER = 1

class BaseLLM(ABC):

    LLM_REGISTRY = {
        LLM_TYPE.SENTENCE_TRANSFORMER : 'sentence-transformers/all-MiniLM-L6-v2'
    }

    def __init__(self,
                 llm_type : LLM_TYPE,
                 device : Optional[torch.device] = torch.device('cpu')
                 ) -> None:
        
        self.device = device
        self.model_str = BaseLLM.LLM_REGISTRY[llm_type]
        # load tokenizer and model pair
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        self.model = AutoModel.from_pretrained(self.model_str).to(self.device)

    @staticmethod
    def create_llm_model(
        llm_type_str : str, 
        device : Optional[torch.device]=torch.device('cpu')):
        """
        Args:
            llm_type_str (str): the type of llm model to be loaded.
            device (torch.device, optional): cpu/cuda/mps. Defaults to torch.device('cpu').
        """
        if llm_type_str.lower() == "sentence_transformer":
            return SentenceTransformer(LLM_TYPE.SENTENCE_TRANSFORMER, device)
        else:
            raise Exception(f"Invalid llm type {llm_type_str}.")
    
    @property
    def model_name(self):
        return self.model_str
    
    @abstractmethod
    def encode(self, sentences, **kwargs):
        pass

class SentenceTransformer(BaseLLM):

    def encode(self, sentences, **kwargs):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        attention_mask = encoded_input['attention_mask']
        # run transformer
        with torch.no_grad(): # just to be safe
            output = self.model(**encoded_input)
        token_embeddings = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        
            





        




