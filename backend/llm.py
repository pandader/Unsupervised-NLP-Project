from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer:

    def __init__(self,
                 model_str : Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
                 device : Optional[torch.device] = torch.device('cpu')
                 ) -> None:
        
        self.device = device
        self.model_str = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        self.model = AutoModel.from_pretrained(self.model_str).to(self.device)

    @property
    def model_name(self):
        return self.model_str
    

    def encode(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        attention_mask = encoded_input['attention_mask']
        # run transformer
        with torch.no_grad(): # just to be safe
            output = self.model(**encoded_input)
        token_embeddings = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)





        




