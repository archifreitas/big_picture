# Imports
import nltk
import re
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# from sklearn.feature_extraction.text import TfidfVectorizer # in case vectorizer is added here 
# import numpy as np # in case this is needed
import pandas as pd

# Arguments to change based on data location and filename
# Data for loading .csv
REL_PATH_INPUT = "../raw_data/all_the_news/"

# Main function
def pre_process(df, source='web', params=None, sample=None, printed=False):
    """
    Main function to pre-process data. 

    Parameters
        ----------
        df : DataFrame
            DataFrame containing the data to pre_process.

        source : string
            Describes the source of data to be pre_processed.
            The default is 'web. Can also be used 'prepared'.
        
        params : dict
            A dictionary contain some or all of the following keywords:
                - 'cat_mapping': True,
                - 'count_numbers': True,
                - 'check_emotions': True,
                - 'vocab_richness': True,
                - 'remove_digits': True,
                - 'remove_punctuation': True,
                - 'remove_emojis': True,
                - 'tokenize': True,
                - 'stopwords': True,
                - 'lemmatize': True,
                - 'stemming': False
        
        sample : float
            Default value is None. Returns a sample of the whole dataset.

        printed : boolean
            Turn on to receive information on the pre_processing data process.
    """
    # Column name for pre_processed data
    pre_processed_text = "pre_processed_text"
    
     
    # For sampling
    if sample:
        df = df.sample(sample)
    
    pp_dict = {'cat_mapping': True,
               'count_numbers': True,
               'check_emotions': True,
               'vocab_richness': True,
               'remove_digits': True,
               'remove_punctuation': True,
               'remove_emojis': True,
               'tokenize': True,
               'stopwords': True,
               'lemmatize': True,
               'stemming': False}
    
    print(params)
    if params:
        for key, val in params.items():
            pp_dict[key] = val
    
    # Prep columns for data pre_processing
    df = data_prep(df, source, printed)
    df['minor_preprocessing'] = df[pre_processed_text]
    df = pp_cat_mapping(df, printed, execute=pp_dict['cat_mapping'])
    df = df.dropna(subset=[pre_processed_text]).reset_index(drop=True)
    
    if printed:
        print("-------------------------")
        print("Minor pre-processing done")
        print("-------------------------")
    
    # Add new columns for label and other features
    df = pp_count_numbers(df, printed, execute=pp_dict['count_numbers'])
    df = pp_check_emotions(df, printed, execute=pp_dict['check_emotions'])
    df = pp_vocab_richness(df, printed, execute=pp_dict['vocab_richness'])
    
    if printed:
        print("-------------------------")
        print("New features added")
        print("-------------------------")
    
    # Remove excess data
    df = pp_remove_digits(df, printed, execute=pp_dict['remove_digits'])
    df = pp_remove_punctuation(df, printed, execute=pp_dict['remove_punctuation'])
    df = pp_remove_emojis(df, printed, execute=pp_dict['remove_emojis'])
    
    if printed:
        print("-------------------------")
        print("Excess data removed")
        print("-------------------------")

    
    # Divide data and pre_process
    df = pp_tokenize(df, printed, execute=pp_dict['tokenize'])
    df = pp_stopwords(df, printed, execute=pp_dict['stopwords'])
    df = pp_lemmatizing(df, printed, execute=pp_dict['lemmatize'])
    df = pp_stemming(df, printed, execute=pp_dict['stemming'])
    
    if printed:
        print("-------------------------")
        print("Data pre-processed")
        print("-------------------------")

    return df

# Data cleaning
def data_prep(df, source, printed=False):
    
    if printed:
        print('Preparing pre_processed_text column')
    
    if source == 'web':
        CONTENT_COL = "content"
        DESCRIPTION_COL = "short_description"
        HEADLINE_COL = "headline"
        
        
        df = df.drop(columns=['Unnamed: 0', 'index'])
        df = df[df['content'] != "Invalid file"].reset_index(drop=True)
        df[CONTENT_COL] = df[CONTENT_COL].replace(['\n','\r'],' ', regex=True)
        df['pre_processed_text'] = df[HEADLINE_COL] + " " + df[DESCRIPTION_COL] + " " + df[CONTENT_COL]

        
        return df
    
    elif source == 'prepared':
        HEADLINE_COL = "title"
        CONTENT_COL = "content"
        
        df[CONTENT_COL] = df[CONTENT_COL].replace(['\n','\r'],' ', regex=True)
        df['pre_processed_text'] = df[HEADLINE_COL] + " " + df[CONTENT_COL]
        
        return df

def pp_cat_mapping(df, printed=False, execute=False):

    my_dict = {'CRIME': 'Crime',
               'ENTERTAINMENT': 'Entertainment',
               'WORLD NEWS': 'World News',
               'IMPACT': 'Other',
               'POLITICS': 'Politics',
               'WEIRD NEWS': 'Other',
               'BLACK VOICES': 'Activism',
               'WOMEN': 'Entertainment',
               'COMEDY': 'Entertainment',
               'QUEER VOICES': 'Activism',
               'SPORTS': 'Sports',
               'BUSINESS': 'Business',
               'TRAVEL': 'Culture',
               'MEDIA': 'Media',
               'TECH': 'Technology',
               'RELIGION': 'Religion',
               'SCIENCE': 'Science',
               'LATINO VOICES': 'Activism',
               'EDUCATION': 'Education',
               'COLLEGE': 'Education',
               'PARENTS': 'Other',
               'ARTS & CULTURE': 'Culture',
               'STYLE': 'Trends',
               'GREEN': 'Activism',
               'TASTE': 'Culture',
               'HEALTHY LIVING': 'Health',
               'THE WORLDPOST': 'World News',
               'GOOD NEWS': 'Other',
               'WORLDPOST': 'World News',
               'FIFTY': 'Other',
               'ARTS': 'Culture',
               'WELLNESS': 'Health',
               'PARENTING': 'Other',
               'HOME & LIVING': 'Trends',
               'STYLE & BEAUTY': 'Trends',
               'DIVORCE': 'Other',
               'WEDDINGS': 'Other',
               'FOOD & DRINK': 'Culture',
               'MONEY': 'Other',
               'ENVIRONMENT': 'Activism',
               'CULTURE & ARTS': 'Culture'}
    
    if execute:
        df['label'] = df.category.map(lambda x: my_dict[x])

        if printed:
            print('Mapping labels')

        return df
    return df    

# Add features
def pp_count_numbers(df, printed=False, execute=False):
    '''Create number of "news + headline" decimals column'''
    
    if execute:
        df['nrs_count'] = df['pre_processed_text'].str.count('[+-]?([0-9]*[.])?[0-9]+')
        df['nrs_count'] = df['nrs_count'].fillna(0)
        df['nrs_count'] = df['nrs_count'].astype(float).astype(int)

        if printed:
            print('Counting numbers')

        return df
    return df

def pp_check_emotions(df, printed=False, execute=False):
    '''Counts types of entonation'''
    
    if execute:
        df['questions'] = df['pre_processed_text'].str.count('\?')
        df['exclamations'] = df['pre_processed_text'].str.count('\!')
        df['irony'] = df['pre_processed_text'].map(lambda x: len(re.findall('\?!|\!\?',x)))

        if printed:
            print('Revealing emotions')

        return df
    return df  

def pp_vocab_richness(df, printed=False, execute=False):

    def vocab_richness(text):
        tokens = word_tokenize(text)
        total_length = len(tokens)
        unique_words = set(tokens)
        unique_word_length = len(unique_words)
       
        if total_length > 0:
            return unique_word_length / total_length
        return 0
    
    if execute:
        df['vocab_richness'] = df['pre_processed_text'].apply(lambda x: vocab_richness(x))

        if printed:
            print('Analyzing vocabulary bank account!')

        return df
    return df

# Pre_process data
def pp_tokenize(df, printed=False, execute=False):
    
    if execute:
        df['pre_processed_text'] = df['pre_processed_text'].apply(lambda x: word_tokenize(x))

        if printed:
            print('Tokenizing')

        return df
    return df

def pp_stopwords(df, printed=False, execute=False, language='english'):
    
    if execute:
        stop_words = set(stopwords.words(language))
        df['pre_processed_text'] = df['pre_processed_text']\
                                .apply(lambda x: [word for word in x if not word in stop_words])

        if printed:
            print('Checking for stopwords')

        return df
    return df  

def pp_stemming(df, printed=False, execute=False, language='english'):
    
    if execute:
        stemmer = SnowballStemmer(language=language)
        
        df['pre_processed_text'] = df['pre_processed_text']\
                                    .apply(lambda x: [stemmer.stem(word) for word in x])

        if printed:
            print('Generating branches')

        return df
    return df

def pp_lemmatizing(df, printed=False, execute=False, upgrade=False):
    
    def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)
    
    if execute:
        lemmatizer = WordNetLemmatizer()
        
        if upgrade:
            nltk.download('averaged_perceptron_tagger')
            
        df['pre_processed_text'] = df['pre_processed_text']\
                                    .map(lambda x: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x])

        if printed:
            print('Lematizing, yummy!')

        return df
    return df

# Remove excess data
def pp_remove_digits(df, printed=False, execute=False):
    '''Removes all the digits from the string'''
    
    if execute:
        df['pre_processed_text'] = df['pre_processed_text'].str.replace('[+-]?([0-9]*[.])?[0-9]+', '', regex=True)

        if printed:
            print('Removing digits')

        return df
    return df 

def pp_remove_punctuation(df, printed=False, execute=False):
 
    if execute:
        real_string_punctuation = string.punctuation + "—" + '”' + "’" + "‘" + "…" + '“' + '´' + "`" + "«" + "»"
        df['pre_processed_text'] = df['pre_processed_text'].apply(lambda x: ''\
                                            .join(word for word in x if word not in real_string_punctuation))

        if printed:
            print('Removing pesky dots')

        return df
    return df

def pp_remove_emojis(df, printed=False, execute=False):

    def deEmojify(text):
            regrex_pattern = re.compile(pattern = "["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                                "]+", flags = re.UNICODE)
            return regrex_pattern.sub(r'',text)
    
    if execute:
        df['pre_processed_text'] = df['pre_processed_text'].apply(lambda x: deEmojify(x))

        if printed:
            print('Disabling emojis')

        return df
    return df 

if __name__ == "__main__":
    pass
