# Imports
import nltk
import re # regex
import string 
from nltk.corpus import stopwords # remove stopwords
from nltk.tokenize import word_tokenize # tokenizing
from nltk.stem.snowball import SnowballStemmer # stemming (improved version of PorterStemmer)(optional)
from nltk.stem import WordNetLemmatizer # lematizing with POS tags (optional)
from nltk.corpus import wordnet
from get_merged_data import get_data
# from sklearn.feature_extraction.text import TfidfVectorizer # in case vectorizer is added here 
# import numpy as np # in case this is needed
import pandas as pd

# Arguments to change based on data location and filename
# Data for loading .csv
REL_PATH_INPUT = "../raw_data/data_12k/"

# Relevant Columns in .csv to change based on data source
CONTENT_COL = "content"
DESCRIPTION_COL = "short_description"
HEADLINE_COL = "headline"

# Parameters for pre-processing methods
# If "all_true" is set to true, we will lematize
args_methods = {"all_true":True, 
                "lower_case":True, 
                "no_digits":True,
                "rem_punct":True,
                "rem_stopwords":True,
                "stem_not_lematize":True,
                "lematize":False}

# Pre-process:
# 1 Read Raw data, basic clean data, create base news+headline column and other basic support data columns
def pre_process(**kwargs):
    # column where all "news" content is stored
    news_all_data = "news_all_data"

    df =  get_data(REL_PATH_INPUT)
    df = df.sample(150)
    df[CONTENT_COL] = df[CONTENT_COL].replace('\n',' ', regex=True)

    print("read_csv")
    
    # Drop NA's and drop columns where there's only the string "Invalid file"
    df[news_all_data] = df[CONTENT_COL] + " " + df[DESCRIPTION_COL] + " " + df[HEADLINE_COL]
    df = df.dropna(subset=[news_all_data]).reset_index()
    df = df[df[news_all_data] != "Invalid file"].reset_index(drop=True)

    print("read_csv")

    #import ipdb; ipdb.set_trace()
    # 1.1 lowercase "news + headline" column
    if kwargs.get("lower_case") or kwargs.get("all_true"): 
        df[news_all_data] = df[news_all_data].str.lower()

    print("lowercase")

    # 1.2 Create number of "news + headline" decimals column (we need to find a way to use this in the future)
    df['nrs_count'] = df[news_all_data].str.count('\d')
    df['nrs_count'] = df['nrs_count'].fillna(0)
    df['nrs_count'] = df['nrs_count'].astype(float).astype(int)

    print("nrs count")

    # 1.3 remove digits from news_all_data column
    if kwargs.get("no_digits") or kwargs.get("all_true"):
        df[news_all_data] = df[news_all_data].apply(lambda x: ''.join(word for word in x if not word.isdigit()))

        print("remove_digits")

    # 1.4 Create support data columns for emotions (we need to find a way to use this column in the future)
    df['questions'] = df[news_all_data].str.count('\?')
    df['exclamations'] = df[news_all_data].str.count('\!')
    df['irony'] = df[news_all_data].map(lambda x: len(re.findall('\?!|\!\?',str(x))))

    print("emotions")

    # 1.5 Remove punctuation
    if kwargs.get("rem_punct") or kwargs.get("all_true"):
        real_string_punctuation = string.punctuation + "—" + '”' + "’" + '“' + '´' + "`" + "«" + "»"
        df[news_all_data] = df[news_all_data].apply(lambda x: ''\
                                            .join(word for word in x if word not in real_string_punctuation))

        print("punctuation")

    # 1.6 Tokenize
    df[news_all_data] = df[news_all_data].apply(lambda x: word_tokenize(x))

    print("tokenize")

    # 1.7 Remove stopwords
    if kwargs.get("rem_stopwords") or kwargs.get("all_true"):
        stop_words = set(stopwords.words('english'))
        df[news_all_data] = df[news_all_data]\
                                .apply(lambda x: [word for word in x if not word in stop_words])

        print("stopwords")

    # 1.8a Stemming (optional)
    if kwargs.get("stem_not_lematize"):
        stemmer = SnowballStemmer(language='english')
        
        df[news_all_data] = df[news_all_data]\
                                    .apply(lambda x: [stemmer.stem(word) for word in x])

        print("stemming")

    # 1.8b Lematizing with POS tags in english (optional)
    if kwargs.get("lematize") or (kwargs.get("all_true") and kwargs.get("stem_not_lematize")==False):
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        lemmatizer = WordNetLemmatizer()
        df[news_all_data] = df[news_all_data]\
                                    .map(lambda x: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x])

        print("lematizing")

   # 1.9 Adding vocab richness column (we need to find a way to use this column in the future)
    def vocab_richness(text):
        tokens = word_tokenize(text)
        total_length = len(tokens)
        unique_words = set(tokens)
        unique_word_length = len(unique_words)
    
        return unique_word_length / total_length
    
    df[news_all_data] = df[news_all_data].map(lambda x: ' '.join(x))
    df['vocab richness'] = df[news_all_data].apply(lambda x: vocab_richness(x))

    print("vocab_richness")

    return df[news_all_data]

if __name__ == "__main__":
    print(pre_process(**args_methods))