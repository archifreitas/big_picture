"""
Models to cluster seuqences of articles by topic.
"""

import numpy as np
import matplotlib.pyplot as plt

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

    wordcloud : wordcloud object
        Wordcloud object ready to be shown with matplotlib

    model_name : string (default: kmeans)
        Type of model used for clustering
    """

    def __init__(self, cluster, topic, wordcloud):
        self.df = cluster
        self.topic = topic
        self.wordcloud = wordcloud
    
    def show_wordcloud(self, size=8):
        """
        Shows wordcloud using matplotlib
        """
        plt.imshow(self.wordcloud)
        plt.tight_layout(pad = 0)
        plt.show()