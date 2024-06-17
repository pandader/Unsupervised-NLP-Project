import re
import platform
import numpy as np
from typing import List
# nlp
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
# torch
import torch
from torch import Tensor

# device
def get_device() -> torch.device:
    if platform.system().upper() == "DARWIN":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif platform.system().upper() == "WINDOWS":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# compute cos sim between two set of documents 
def doc_cos_sim(doc_emb_a : Tensor, doc_emb_b : Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    return matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a_norm = torch.nn.functional.normalize(doc_emb_a.unsqueeze(0) if len(doc_emb_a.shape) == 1 else doc_emb_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(doc_emb_b.unsqueeze(0) if len(doc_emb_b.shape) == 1 else doc_emb_b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# simple nlp preprocess
def preprocess_text(documents: np.ndarray) -> List[str]:
    """ Basic preprocessing of text
    Steps:
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
    """
    cleaned_documents = [doc.replace("\n", " ") for doc in documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
    cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
    return cleaned_documents

# tokenize and remove stop words
def tokenize_and_remove_stop_words(text: str):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return [word for word in simple_preprocess(str(text), deacc=True) if word not in stop_words]