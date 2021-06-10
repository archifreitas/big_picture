# Imports
import nltk
import re # regex
import string 
from nltk.corpus import stopwords # remove stopwords
from nltk.tokenize import word_tokenize # tokenizing
from nltk.stem.snowball import SnowballStemmer # stemming (improved version of PorterStemmer)(optional)
from nltk.stem import WordNetLemmatizer # lematizing with POS tags (optional)
from nltk.corpus import wordnet

# from sklearn.feature_extraction.text import TfidfVectorizer # in case vectorizer is added here 
# import numpy as np # in case this is needed
import pandas as pd

# Arguments to change based on data location and filename
# Data for loading .csv
REL_PATH_INPUT = "../raw_data/data_30k/"

# Relevant Columns in .csv to change based on data source
CONTENT_COL = "content"
DESCRIPTION_COL = "short_description"
HEADLINE_COL = "headline"

# Parameters for pre-processing methods
# If "all_true" is set to true, we will lematize

# Pre-process:
# 1 Read Raw data, basic clean data, create base news+headline column and other basic support data columns
def pre_process(df,sample=None,
                all_true=False, 
                lower_case=False, 
                no_digits=False,
                rem_punct=False,
                rem_stopwords=False,
                stem_not_lematize=False,
                lematize=False):
    # column where all "news" content is stored
    news_all_data = "news_all_data"

    if sample:
        df = df.sample(sample)

    df[CONTENT_COL] = df[CONTENT_COL].replace('\n',' ', regex=True)

    print("read_csv")
    
    # Drop NA's and drop columns where there's only the string "Invalid file"
    df[news_all_data] = df[CONTENT_COL] + " " + df[DESCRIPTION_COL] + " " + df[HEADLINE_COL]
    df = df.dropna(subset=[news_all_data]).reset_index()
    df = df[df['content'] != "Invalid file"].reset_index(drop=True)

    print("read_csv")

    #import ipdb; ipdb.set_trace()
    # 1.1 lowercase "news + headline" column
    if lower_case or all_true: 
        df[news_all_data] = df[news_all_data].str.lower()

    print("lowercase")

    # 1.2 Create number of "news + headline" decimals column (we need to find a way to use this in the future)
    df['nrs_count'] = df[news_all_data].str.count('\d')
    df['nrs_count'] = df['nrs_count'].fillna(0)
    df['nrs_count'] = df['nrs_count'].astype(float).astype(int)

    # print("nrs count")

    # 1.3 remove digits from news_all_data column
    if no_digits or all_true:
        df[news_all_data] = df[news_all_data].apply(lambda x: ''.join(word for word in x if not word.isdigit()))

        print("remove_digits")

    # 1.4 Create support data columns for emotions (we need to find a way to use this column in the future)
    df['questions'] = df[news_all_data].str.count('\?')
    df['exclamations'] = df[news_all_data].str.count('\!')
    df['irony'] = df[news_all_data].map(lambda x: len(re.findall('\?!|\!\?',str(x))))

    # print("emotions")

    # 1.5 Remove punctuation
    if rem_punct or all_true:
        real_string_punctuation = string.punctuation + "—" + '”' + "’" + '“' + '´' + "`" + "«" + "»"
        df[news_all_data] = df[news_all_data].apply(lambda x: ''\
                                            .join(word for word in x if word not in real_string_punctuation))

        print("punctuation")

    # 1.6 Tokenize
    df[news_all_data] = df[news_all_data].apply(lambda x: word_tokenize(x))

    # print("tokenize")

    # 1.7 Remove stopwords
    if rem_stopwords or all_true:
        stop_words = set(stopwords.words('english'))
        df[news_all_data] = df[news_all_data]\
                                .apply(lambda x: [word for word in x if not word in stop_words])

        print("stopwords")

    # 1.8a Stemming (optional)
    if stem_not_lematize:
        stemmer = SnowballStemmer(language='english')
        
        df[news_all_data] = df[news_all_data]\
                                    .apply(lambda x: [stemmer.stem(word) for word in x])

        print("stemming")

    # 1.8b Lematizing with POS tags in english (optional)
    if lematize or (all_true and not stem_not_lematize):
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        lemmatizer = WordNetLemmatizer()
        nltk.download('averaged_perceptron_tagger')
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

    return df

if __name__ == "__main__":
    from big_picture.get_merged_data import get_data

    df =  get_data(REL_PATH_INPUT)
    
    df1 = pre_process(df,
                all_true=True, 
                lower_case=False, 
                no_digits=False,
                rem_punct=False,
                rem_stopwords=False,
                stem_not_lematize=False,
                lematize=False)

    print(df1.news_all_data.head(5))

    df1.to_csv(r'/data/data_30k_all_true.csv', index = False, header=True)
