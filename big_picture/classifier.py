"""
Label classifier
"""
# Internal libraries
from big_picture.pre_processor import pre_process
from big_picture.vectorizers import embedding_strings, tf_idf
from big_picture.label import Label

# General libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import io
import gcsfs
import torch

# Encoding libraries
from sklearn.preprocessing import OneHotEncoder

# Modelling libraries
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.nn import softmax

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
        self.labels_tag = None
        self.input_shape = None
        self.output_shape = None

    def fit(self, train, model='dropout', source='web', params=None, sample=None, printed=False):
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

        # Train classifier with train data
        ohe = OneHotEncoder()

        params = {
            'lemmatize': False,
        }

        train = pre_process(
            train, 
            source=source, 
            params=params, 
            sample=sample, 
            printed=printed)

        X = embedding_strings(train['minor_preprocessing'])
        y = ohe.fit_transform(train[['label']]).toarray()

        # Save tags for labels to class
        self.labels_tag = ohe.categories_[0]
        
        # Save model variable to class
        es = EarlyStopping(patience=10)

        if model == 'dropout':
            self.input_shape = X.shape[1]
            self.output_shape = y.shape[1]
            self.model = initialize_class_bert_dropout(self.input_shape, self.output_shape)

    
        self.model.fit(
                    X,
                    y,
                    epochs=20,
                    validation_split=0.25,
                    batch_size=32,
                    callbacks=[es],
                    verbose=1
                    )

    def save(self, path):
        '''Saves a classifying model'''
        mdl_path = os.path.join(path, 'model.pkl')
        state_path = os.path.join(path, 'state.pkl')
        
        os.makedirs(path)

        if self.model != None:
            self.model.save_weights(mdl_path)
        else:
            raise Exception('Please fit a model first')

        state = {
            'labels': self.labels,
            'labels_tag': self.labels_tag,
            'threshold': self.threshold,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape
        }

        with open(state_path, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, path):
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else: return super().find_class(module, name)

        mdl_path = os.path.join(path, 'model.pkl')
        state_path = os.path.join(path, 'state.pkl')


        fs = gcsfs.GCSFileSystem(project = 'wagon-bootcamp-311206')
        fs.ls('big_picture_model')
        with fs.open('big_picture_model/model/state.pkl', 'rb') as file:
            state = CPU_Unpickler(file).load()

        self.labels = state['labels'] 
        self.labels_tag = state['labels_tag'] 
        self.threshold = state['threshold'] 
        self.input_shape = state['input_shape'] 
        self.output_shape = state['output_shape'] 
    
        self._init()

        self.model = initialize_class_bert_dropout(self.input_shape, self.output_shape)
        self.model.load_weights(mdl_path)

        #self._init()
        
    def _init(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sa_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def divide_labels(self, world, source='web', params=None, sample=None, printed=False):
        '''
        Populates the classifier with data for clustering.

        Parameters
        ----------
        world : df
            DataFrame to predict labels from and generate world of clusters.
        '''
        if self.model != None:

            # Pre-process data
            world = pre_process(
                world, 
                source=source, 
                params=params, 
                sample=sample, 
                printed=printed)
            
            # Set data to predict
            X = embedding_strings(world['minor_preprocessing'])

            # Predict data
            results = self.model.predict(X)

            # print('results')
            # print(results)

            # Divide into labels
            labels = {key: [] for key in labels_dict.keys()}

            for i, result in enumerate(results):
                for j, label_pred in enumerate(result):
                    if label_pred >= self.threshold:
                        labels[j].append(i)

            # print(labels)

            # Transform into Label() instances                
            self.labels = {}
            self._init()

            for key, value in labels.items():
                print(key, value)
                if value:
                    self.labels[self.labels_tag[key]] = Label(
                                                        world.iloc[value, :].reset_index(),
                                                        self.labels_tag[key],
                                                        tokenizer=self.tokenizer,
                                                        sa_model=self.sa_model
                                                        )
        else:
            raise Exception('Please fit a model first')
    

    def predict(self, df, source='prepared', params=None, sample=None, printed=False):

        # Pre-process data
        df = pre_process(
            df, 
            source=source, 
            params=params, 
            sample=sample, 
            printed=printed)
            
        # Set data to predict
        X = embedding_strings(df['minor_preprocessing'])

        prediction = self.model.predict(X)

        print(prediction)
        # Put into correct labels

        labels = []

        for i, result in enumerate(prediction):
            for j, label_pred in enumerate(result):
                if label_pred >= self.threshold:
                    print(j)
                    labels.append(self.labels_tag[j])
        
        print(labels)
        self._init()

        sa = softmax(self.sa_model(self.tokenizer(
                    df['minor_preprocessing'].iloc[0], 
                    return_tensors='tf',
                    padding=True,
                    max_length=500, #!!!!!!!!!!!!!!!!might need to change
                    truncation=True
                    )).logits).numpy()

        output_df = df[['title', 'url', 'publishedAt', 'author', 'source']]
        output_df[['SA']] = sa[0][1]-sa[0][0]

        output = {}
        # Check if it is embedded X
        for label in labels:
            cluster = self.labels[label].predict(X)
            output[label] = self.labels[label].clusters[cluster]
            output[label].df = pd.concat([output_df,output[label].df],axis=0).drop_duplicates('url')

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


def initialize_class_bert_dropout(shape, output):
    
    model = models.Sequential()
    
    model.add(layers.Dense(700, activation='relu', input_dim=shape))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(50, activation='relu'))
    
    model.add(layers.Dense(output, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
