import re
import random
import numpy as np
import pandas as pd
from typing import List, Optional, Mapping, Tuple
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# local
from .utils import preprocess_text

def top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    indices = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    return np.array(indices)

def top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    return np.array(top_values)

def extract_words_per_topic(
          words: List[str],
          documents: pd.DataFrame,
          c_tf_idf: csr_matrix,
          top_n_words : Optional[int]=5) -> Mapping[str, List[Tuple[str, float]]]:

    """ Based on tf_idf scores per topic, extract the top n words per topic

    If the top words per topic need to be extracted, then only the `words` parameter
    needs to be passed. If the top words per topic in a specific timestamp, then it
    is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
    labels.

    Arguments:
        words: List of all words (sorted according to tf_idf matrix position)
        documents: DataFrame with documents and their topic IDs
        c_tf_idf: A c-TF-IDF matrix from which to calculate the top words

    Returns:
        topics: The top words per topic
    """
    labels = sorted(list(documents.Topic.unique()))
    labels = [int(label) for label in labels]
    indices = top_n_idx_sparse(c_tf_idf, top_n_words)
    scores = top_n_values_sparse(c_tf_idf, indices)
    sorted_indices = np.argsort(scores, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(scores, sorted_indices, axis=1)
    topics = {label: [(words[word_index], score)
                        if word_index is not None and score > 0
                        else ("", 0.00001)
                        for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                        ] for index, label in enumerate(labels)}
    return topics

### c-tfidf
class ClassTfidfTransformer(TfidfTransformer):
    
    def __init__(self, 
                 bm25_weighting: bool = False, 
                 reduce_frequent_words: bool = False,
                 seed_words: List[str] = None,
                 seed_multiplier: float = 2):
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.seed_words = seed_words
        self.seed_multiplier = seed_multiplier
        super(ClassTfidfTransformer, self).__init__()

    def fit(self, X: sp.csr_matrix, multiplier: np.ndarray = None):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            multiplier: A multiplier for increasing/decreasing certain IDF scores
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape

            # Calculate the frequency of words across all classes
            df = np.squeeze(np.asarray(X.sum(axis=0)))

            # Calculate the average number of samples as regularization
            avg_nr_samples = int(X.sum(axis=1).mean())

            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1+((avg_nr_samples - df + 0.5) / (df+0.5)))

            # Divide the average number of samples by the word frequency
            # +1 is added to force values to be positive
            else:
                idf = np.log((avg_nr_samples / df)+1)

            # Multiplier to increase/decrease certain idf scores
            if multiplier is not None:
                idf = idf * multiplier

            self._idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self

    def transform(self, X: sp.csr_matrix):
        """Transform a count-based matrix to c-TF-IDF

        Arguments:
            X (sparse matrix): A matrix of term/token counts.

        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm='l1', copy=False)

            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)

            X = X * self._idf_diag

        return X


### keyword extractor
class KeywordExtractor:

    def __init__(self,
                 df_assigned : pd.DataFrame,
                 df_not_assigned: pd.DataFrame,
                 top_n_words : Optional[int]=5):
        # combined
        self.top_n_words = top_n_words
        self.df_comb = pd.concat([df_assigned, df_not_assigned], axis=0).reset_index(drop=True)
        self.df_comb.label_id = self.df_comb.index        
        concat_text = self.df_comb.text.apply(lambda x: ' '.join(x))
        self.processed_doc_by_topic = preprocess_text(concat_text)
        # to output
        self.topics = None
        self.df_concat = None

    def run(self):
        # get count vector representation
        vectorizer_model = CountVectorizer(ngram_range=(1, 3))
        count_rep = vectorizer_model.fit_transform(self.processed_doc_by_topic)
        words = vectorizer_model.get_feature_names_out()
        # ctfidf
        ctfidf_model = ClassTfidfTransformer()      
        ctfidf_model = ctfidf_model.fit(count_rep)
        ctfidf = ctfidf_model.transform(count_rep)
        # generate topics
        df_concat = pd.DataFrame(columns=['Document', 'Topic'])
        df_concat['Document'] = self.processed_doc_by_topic
        df_concat['Topic'] = self.df_comb.label_id
        self.topics = extract_words_per_topic(words, df_concat, ctfidf, self.top_n_words)

    def get_topics_and_docs(self, df_data, num_reps):
        # very basic version, just random sample [can be improve]
        content = []
        for idx, row in self.df_comb.iterrows():
            num_choices = min(len(row['id']), num_reps)
            indices = random.choices(row['id'], k=num_choices)
            docs = df_data.loc[indices]['text'].values.tolist()
            kws = ', '.join([each[0] for each in self.topics[idx]])
            content.append([kws, docs])
        return pd.DataFrame(content, columns=["keywords", "docs"])
    



    
    