import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#import hdbscan as hdb
from sklearn.cluster import KMeans, Birch
from wordcloud import WordCloud

from big_picture.clusters import Cluster
from big_picture.pre_processor import pre_process
from big_picture.vectorizers import tf_idf, embedding_strings

class Label():
    """
    Class that creates an object to be labelled and organized by topic.

    Parameters
    ----------
    df : df
        DataFrame containing the features to be analyzed:
        - Title;
        - Authors;
        - Publisher;
        - Date;
        - Link;
        - Content.

    label: string
        Label correponding to the topic of the class.

    vec_name: string (default: embedding_string)
        Name of vectorizer to be used for vectorizing. Options:
        embedding_string
        tf_idf
    
    model_name: string (default: kmeans)
        Name of model to be used for clustering
    
    """
    def __init__(self, df, label, vec_name='embedding_strings', model_name='kmeans', **kwargs):
        self.label = label

        if vec_name == 'tf_idf':
            vectors, self.vectorizer = tf_idf(df.pre_processed_text)
        elif vec_name == 'embedding_strings':
            vectors, self.vectorizer = embedding_strings(df.pre_processed_text,  return_model=True)
        else:
            pass
 
        self.model = None
        self.sizes = None
        if model_name == 'kmeans':
            self.clusters= self.kmeans(df, 
                                  'pre_processed_text', 
                                  vectors, 
                                  clusters=1+len(df)//30,
                                  **kwargs
                                  )
        elif model_name == 'birch':
            self.clusters= self.birch(df, 
                                  'pre_processed_text', 
                                  vectors,
                                  clusters=1+len(df)//30,
                                  **kwargs
                                  )
        else:
            print("No model was found, this may cause problems in the future")

    def predict(self, vector):
        """
        Function that predicts the closest cluster number of a given vectorized sample
        """
        if not self.model:
            raise Exception('No model found')
        return self.model.predict(vector)[0]
    
    def extract_top_n_words_per_topic(self, tf_idf, count, docs_per_topic, n=20):
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

    def c_tf_idf(self, documents, m, ngram_range=(1, 1)):
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

    def output_format(self, X, column, **kwargs):
        """
        Returns a list of cluster objects with the dataframe and topic
        Optionally the size of each cluster
        """

        docs_per_topic = X.groupby(['topic'], as_index = False).agg({column: ' '.join})

        # print(X.pre_processed_text.iloc[0])

        # print(X.groupby(['topic'], as_index = False).count())

        tf_idf, count = self.c_tf_idf(docs_per_topic[column].values, m=len(X))

        top_n_words = self.extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

        self.sizes = (X.groupby(['topic'])
                        .content
                        .count()
                        .reset_index()
                        .rename({"topic": "topic", "content": "Size"}, axis='columns')
                        .sort_values("Size", ascending=False))
        

        clusters = []

        for topic in X['topic'].unique():
            clusters.append((X[X.topic == topic]))

        output = []
        for i, cluster in enumerate(clusters):
            wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        min_font_size = 10).generate(docs_per_topic[column].iloc[i])
            output.append(
                Cluster(
                    cluster,
                    top_n_words[i],
                    wordcloud,
                    **kwargs
                )
            )

        return output

    def kmeans(self, X, column, vectors, clusters=8, **kwargs):
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

        clusters : int (default: 8)
            intended number of clusters
        """

        self.model = KMeans(n_clusters=clusters).fit(vectors)

        X['topic'] = self.model.labels_

        return self.output_format(X, column,**kwargs)


    def birch(self, X, column, vectors, clusters, threshold=0.5, **kwargs):
        """
        Birch model that outputs a list of cluster objects with the dataframe and topic

        Parameters
        ----------
        X : df
            Data Frame of articles

        column : string
            the preproccessed column name

        vectors : string
            vectorized data of the preproccessed column

        threshold : int (default: 0.5)
            The radius of the subcluster obtained by merging a new sample and the closest 
            subcluster should be lesser than the threshold. Otherwise a new subcluster is started. 
            Setting this value to be very low promotes splitting and vice-versa.
        """

        self.model = Birch(threshold=threshold, n_clusters=clusters).fit(vectors)

        X['topic'] = self.model.labels_

        return self.output_format(X, column,**kwargs)
