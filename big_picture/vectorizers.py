"""
Machine learning an deep learning vectorizers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(X):
    """
    Vectorize a sequence of strings using tf_idf
    """

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(X)
    return vectors