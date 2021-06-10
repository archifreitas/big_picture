# General libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import requests

# Embedding libraries
from sentence_transformers import SentenceTransformer

# Encoding libraries
from sklearn.preprocessing import OneHotEncoder

# Modelling libraries
from tensorflow.keras import models
from tensorflow.keras import layers


class Classifier():
    """
    Class that creates an object to be labelled by topic.

    Parameters
    ----------
    instance : df
        DataFrame containing the features to be analyzed:
        - Title;
        - Authors;
        - Publisher;
        - Date;
        - Link;
        - Content.

    """
    def __init__(self, instance, label):
        self.df = instance
        self.label = label

def prepare_data_for_classifying(df):
    '''Return the concatenation of strings

    Parameters
    ----------
    df : df
        DataFrame containing the columns to be merged:
        - Title;
        - Content.
    To add more pre-processing if necessary'''
    new_df = df.copy()
    new_df['merged_strings'] = df.title + df.content
    return new_df

def embedding_string(strings, progress_bar=False):
    '''Embedding of the instance string 
    represented by its title and content.
    Note: Receives an array of strings, or a single string'''
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(strings, show_progress_bar=progress_bar)
    return embeddings

def load_model(model='classifier_baseline'):
    '''Load the model to classify the topic of the string.
    
    Parameters
    ----------
    model : string
        Name of the model to be used.
    '''
    file_path = 'models/' + model
    model = models.load_model(file_path)
    return model


def make_a_prediction(model, strings):
    '''Outputs the label of a vectorized string.

    Parameters
    ----------

    model: string
        Name of the model to be used.

    strings : array or series of strings
        Array of strings to be labelled.
    '''
    vectorized_string = embedding_string(strings)
    model = load_model(model)
    return model.predict(vectorized_string)

def classifying_threshold(predictions,threshold):
    '''Labels prediction depending on probability distribution of the classes'''
    val_dict = {0: 'Activism',
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
    labels = []
    for idx, val in enumerate(predictions):
        if val > threshold:
            labels.append(val_dict[idx])
    return labels


def classifying_pipeline(df, model):
    '''Generates an instance of Classifier with the dataframe 
    and label associated with the dataframe.

    Parameters
    ----------
    df : array or series of strings
        Array of strings to be labelled.
    
    model : string
        Name of the model to be used.
    '''
    data = prepare_data_for_classifying(df)
    results = make_a_prediction(data['merged_strings'])

    data['label'] = classifying_threshold(results)

    output = []

    for i in range(len(data)):
        for label in data['label']:
            output.append(Classifier(data, label))

    return output


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


def initialize__class_bert_dropout():
    
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

