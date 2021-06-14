"""
Models to cluster seuqences of articles by topic.
"""

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.nn import softmax
from sklearn.preprocessing import MinMaxScaler

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

    def __init__(self, cluster, topic, wordcloud, **kwargs):
        self.df = cluster[['headline', 'link', 'date']].reset_index().drop(columns='index')
        self.topic = topic
        self.wordcloud = wordcloud

        tokenizer = kwargs.get('tokenizer')
        model = kwargs.get('sa_model')

        texts = list(cluster["news_all_data"])

        encoded_input = tokenizer(texts, 
                            return_tensors='tf',
                            padding=True,
                            max_length=500, #!!!!!!!!!!!!!!!!might need to change
                            truncation=True)
        
        output = model(encoded_input)

        my_array = softmax(output.logits).numpy()

        df = pd.DataFrame(my_array, columns = ['Negative','Positive'])

        df['SA'] = df['Positive'] - df['Negative']

        
        # Optional Scalling (we may find out that news are not mostly negatively biased)
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) # Instanciate StandarScaler
        self.scaler.fit(df[['SA']]) # Fit scaler to data
        df['norm_SA'] = self.scaler.transform(df[['SA']]) # Use scaler to transform data
        self.df = pd.concat([self.df,df],axis=1)
    
    def show_wordcloud(self, size=8):
        """
        Shows wordcloud using matplotlib
        """
        plt.imshow(self.wordcloud)
        plt.tight_layout(pad = 0)
        plt.show()

