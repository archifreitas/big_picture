"""
Label classifier
"""
# Internal libraries
from big_picture.clusters import kmeans
from big_picture.pre_processor import pre_process
from big_picture.vectorizers import embedding_string, tf_idf

# General libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import requests

# Encoding libraries
from sklearn.preprocessing import OneHotEncoder

# Modelling libraries
from tensorflow.keras import models
from tensorflow.keras import layers

# Label dictionary
labels_dict = {0: 'Activism',
               1: 'Business',
               2: 'Crime',
               3: 'Culture',
               4: 'Education',
               5: 'Entertainment',
               6: 'Health',
               7: 'Media',
               8: 'Other',
               9: 'Politics',
               10: 'Religion',
               11: 'Science',
               12: 'Sports',
               13: 'Technology',
               14: 'Trends',
               15: 'World News'}

class Classifier():
    """
    Class that creates an object with a model and the topics associated to the model.

    Parameters
    ----------
    
    labels: class
        Class containing the subsets of the original dataset by label.

    threshold: float
        Value between 0 and 1 for the classifier to consider that a prediction belongs to a certain topic.
    """

    def __init__(self, labels, threshold=0.4):
        self.labels = labels
        self.threshold = threshold
        self.model = None
    
    def classifying_threshold(self, predictions,threshold):
        '''Labels prediction depending on probability distribution of the classes'''
        val_dict = labels_dict
        labels = []
        for idx, val in enumerate(predictions):
            if val > threshold:
                labels.append(val_dict[idx])
        return labels
    
    def fit(train, model=initialize_class_bert_dropout()):
        '''
        Generate a model and fit it to the train_data.

        Parameters
        ----------
        train : df
            DataFrame containing the data to train

        world : df
            DataFrame to predict labels from and generate world of clusters.
        
        model : model
            Classification model to be used.
        '''

        # Pre-process data
        pre_processed_train = pre_process(train,
                                          sample=1,
                                          all_true=True)                        
        
        # Train classifier with train data
        X = pre_processed_train.drop(columns='label')
        y = pre_processed_train.label
        
        # Save model variable to class
        self.model = model.fit(X, y)

    def save():
        if self.model != None:
            filename = 'finalized_model.sav'
            pickle.dump(model, open(filename, 'wb'))
        else:
            raise Exception('Please fit a model first')
    
     # Predict model with all_news_data_dataset
        model.predict()
        # return model and new_world
       
    def divide_labels():

        # Pre-process data
        pre_processed_train = pre_process(train,
                                          sample=1,
                                          all_true=True)

        pre_processed_world = pre_process(world,
                                          sample=1,
                                          all_true=True)   

        pass

        
        ##### Divide into specific labels
        # Divide all_news_data_dataset into 16 Topic() instances
        # return classifier instance



        # Pick model to train here
        model = initialize_class_bert_0()

        pre_processed_df = pre_process(df,
                                    sample=1,
                                    all_true=True)

        X = pre_processed_df.drop(columns='label')
        y = pre_processed_df.label

        model.fit()

        #### Add preprocessing and divide in topics
        #### Output should be a dictionary of Topic() instances
        
        output = []

        # for i in range(len(data)):
        #     for label in data['label']:
        #         output.append(Topic(data[data['label'] == label], label))

        classifier = Classifier(model, output)
        pikle.dump()

    
    def predict(self, string):
        predictions = self.model.predict(embedding_string(string))
        return self.classifying_threshold(predictions, self.threshold)


# Save pipeline function



### Classification reports

def summary_report():
    pass


def summary_confusion_matrix():
    pass

### Models in use

def initialize_class_bert_0():
    '''Initial classifier using BERT enconding with sentence transformer'''
    
    model = models.Sequential()
    
    model.add(layers.Dense(300, activation='relu', input_dim=embeddings.shape[1]))
    
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    
    model.add(layers.Dense(16, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def initialize_class_bert_dropout():
    
    model = models.Sequential()
    
    model.add(layers.Dense(300, activation='relu', input_dim=embeddings_200k.shape[1]))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(50, activation='relu'))
    
    model.add(layers.Dense(16, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

