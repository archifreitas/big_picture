
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
    """
    def __init__(self, df, label, vec_name, model_name):
        self.label = label

        pre_processed_df = pre_process(df,
                                sample=1,
                                all_true=True)

        if vec_name == 'tf_idf':
            self.vectors, self.vectorizer = tf_idf(pre_processed_df.news_all_data)
        else:
            pass



        if model_name == 'kmeans':
            self.clusters= kmeans(pre_processed_df, 
                                  'news_all_data', 
                                  vectors, 
                                  clusters=8)
        else:
            pass