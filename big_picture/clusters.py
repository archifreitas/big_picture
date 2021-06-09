"""
Models to cluster seuqences of articles by topic.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan as hdb
from sklearn.cluster import KMeans

class Cluster():
    """
    Class that creates an object for a single cluster of data. 
    Stores its dataframe and toic words describing it

    Parameters
    ----------
    cluster : df
        DataFrame of an individual cluster

    topic : list
        List of words describing cluster
    """
    def __init__(self, cluster, topic):
        self.df = cluster
        self.topic = topic
    

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    """
    Extracts the top words of a topics.
    Takes a dataframe with aggregated articles grouped by topic 
    and the information ontained through tf-idf vectorization of it.
    """
    words = count.get_feature_names()
    labels = list(docs_per_topic.topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = [[words[j] for j in indices[i]][::-1] for i, label in enumerate(labels)]
    return top_n_words

def extract_cluster_sizes(df):
    """
    Extracts the cluster sizes of each cluster.
    Takes a dataframe with a column 'topic' asigning a topic number to each cluster.
    """
    topic_sizes = (df.groupby(['topic'])
                     .content
                     .count()
                     .reset_index()
                     .rename({"topic": "topic", "content": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    Vectorizer a dataframe of documents that have been agregated by cluster.
    Parameter 'm' is the total number of articles in the data set
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def output_format(X, column, return_cluster_sizes):
    """
    Returns a list of cluster objects with the dataframe and topic
    Optionally the size of each cluster
    """

    docs_per_topic = X.groupby(['topic'], as_index = False).agg({column: ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic[column].values, m=len(X))

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=10)

    clusters = []

    for topic in X['topic'].unique():
        clusters.append((X[X.topic == topic]))
    
    output = []
    for i, cluster in enumerate(clusters):
        output.append(
            Cluster(cluster,top_n_words[i])
        )

    if return_cluster_sizes:
        return output, extract_cluster_sizes(X)
    return output

def kmeans(X, column, vectors, clusters=8, return_cluster_sizes=False):
    """
    Kmean model that outputs a list of cluster objects with the dataframe and topic

    Parameters
    ----------
    X : df
        Data Frame of articles

    column : string
        the preproccessed column name

    vectors : string
        vectorized data of the preproccessed column

    clusters : int
        intended number of clusters

    return_cluster_sizes : bolean
        Optionally return the size of each cluster
    """

    model = KMeans(n_clusters=clusters).fit(vectors)

    X['topic'] = model.labels_

    return output_format(X,column, return_cluster_sizes=return_cluster_sizes)

def hdbscan(X, column, vectors, min_cluster_size=5, return_cluster_sizes=False):
    """
    Hbdscan clustering model that outputs a list of cluster objects with the dataframe and topic

    Parameters
    ----------
    X : df
        Data Frame of articles

    column : string
        the preproccessed column name

    vectors : string
        vectorized data of the preproccessed column

    min_cluster_size : int
        minimum cluster size of the clustering model
        datapoints can still be classified as outliers (clusters of 1)

    return_cluster_sizes : bolean
        Optionally return the size of each cluster
    """

    model = hdb.HDBSCAN(min_cluster_size=min_cluster_size,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(vectors)

    X['topic'] = model.labels_

    return output_format(X,column, return_cluster_sizes=return_cluster_sizes)