"""
Machine learning and deep learning vectorizers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import umap

# Embedding libraries
from sentence_transformers import SentenceTransformer


def embedding_strings(strings, progress_bar=False):
    '''Embedding of the instance string 
    represented by its title and content.
    Note: Receives an array of strings, or a single string'''
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(strings, show_progress_bar=progress_bar)
    return embeddings

def tf_idf(X):
    """
    Vectorize a sequence of strings using tf_idf
    """

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(X)
    return vectors, vectorizer

def reduce_dimensions(data, dimensions, n_neigbors=15):
    """
    Model that reduces the dimensions of a sequence of vectors.

    Parameters
    ----------
    data : df
        DataFrame of vectorized data

    dimensions : float
        Number of dimensions of the output vectors

    n_neigbors : float (optional, default 15)
    The size of local neighborhood (in terms of number of neighboring
    sample points) used for manifold approximation. Larger values
    result in more global views of the manifold, while smaller
    values result in more local data being preserved. In general
    values should be in the range 2 to 100.
    """
    return umap.UMAP(n_neighbors=n_neigbors, 
                            n_components=dimensions, 
                            metric='cosine').fit_transform(data)