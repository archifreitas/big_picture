"""
Label classifier
"""
# Internal libraries
from big_picture.clusters import kmeans
from big_picture.pre_processor import pre_process
from big_picture.vectorizers import embedding_strings, tf_idf
from big_pciture.label import Label

# General libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Encoding libraries
from sklearn.preprocessing import OneHotEncoder

# Modelling libraries
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Reports libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.model = None
        self.labels = None


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
        ohe = OneHotEncoder()

        X = embedding_strings(pre_processed_train.drop(columns='label'))
        y = ohe.fit_transform(pre_processed_train[['label']].toarray())

        # Save tags for labels to class
        self.labels_tag = ohe.categories_[0]
        
        # Save model variable to class
        es = EarlyStopping(patience=10)

        self.model = model.fit(X,
                               y,
                               epochs=20,
                               validation_split=0.25,
                               batch_size=32,
                               callbacks=[es],
                               verbose=1
                               )


    def save():
        '''Saves a classifying model'''
        if self.model != None:
            filename = 'finalized_model.sav'
            pickle.dump(model, open(filename, 'wb'))
        else:
            raise Exception('Please fit a model first')
       

    def divide_labels(world):
        '''
        Populates the classifier with data for clustering.

        Parameters
        ----------
        world : df
            DataFrame to predict labels from and generate world of clusters.
        '''
        if self.model != None:

            # Pre-process data
            pre_processed_world = pre_process(world,
                                            sample=1,
                                            all_true=True)   
            
            # Set data to predict
            X = embedding_strings(pre_processed_world)

            # Predict data
            results = self.model.predict(X)

            # Divide into labels
            labels = {key: [] for key in labels_dict.keys()}

            for i, result in enumerate(results):
                for j, label_pred in enumerate(result):
                    if label_pred >= self.threshold:
                        labels[j].append(i)


            # Transform into Label() instances                
            self.labels = {}

            for key, value in labels.items():
                self.labels[key] = Label(pre_processed_world.iloc[value, :], self.labels_tag[key])

        else:
            raise Exception('Please fit a model first')
    

    def predict(self, df):

        pre_processed_X = pre_process(df)
        embedded_X = embedding_string(pre_processed_X)
        prediction = self.model.predict(embedded_X)

        # Put into correct labels

        labels = []

        for i, result in enumerate(results):
            for j, label_pred in enumerate(result):
                if label_pred >= self.threshold:
                    labels.append(self.labels_tag[j])
        
        output = []
        # Check if it is embedded X
        for label in labels:
            output.append((label, self.labels[label].predict(embedded_X)))

        return output

    ### Classification reports

    def reports(self, y_true, report=1):
        '''
        Generate a model and fit it to the train_data.

        Parameters
        ----------
        y_true : pd.series
            Panda Series containing the y_true used for training the model.

        report : int
            Default value is 1, which will print the classification report.
            Use 0 to plot the confusion matrix for the different labels.
        '''

        if self.model != None:

            classes = np.argmax(self.model.predict(X), axis=-1)
            val_dict = {}

            for idx, val in enumerate(self.labels_tag):
                val_dict[val] = int(idx)

            y_true = y_true.map(val_dict)    

            # Classification report
            if report:
                return print(classification_report(y_true, self.classes))

            # Confusion Matrix plot
            else:
                cm_array = confusion_matrix(y_true, self.classes)

                df_cm = pd.DataFrame(cm_array, 
                                     index = [i for i in labels],
                                     columns = [i for i in labels])
                plt.figure(figsize = (15,8))
                return sns.heatmap(df_cm, annot=True)

        else:
            raise Exception('Please fit a model first')
        



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
