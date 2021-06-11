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
REL_PATH_INPUT = "../raw_data/all_the_news/"

# Parameters for pre-processing methods
# If "all_true" is set to true, we will lematize

# Pre-process:
# 1 Basic cleaning of data is not optional, create base news+headline column and other basic support data columns
def pre_process(df,
                dataset=None,
                sample=None,
                all_true=False, 
                lower_case=False, 
                no_digits=False,
                rem_punct=False,
                de_emojify=False,
                tokenize=False,
                rem_stopwords=False,
                stem_not_lematize=False,
                lematize=False):
    """
    dataset can either corerspond to "hp" (huffington post .csv(s) dataset) or "all_the_news"
    which corresponds to kaggle all the news dataset
    sample corresponds to sample number for testing purposes
    """
    # column where all "news" content is stored
    news_all_data = "news_all_data"

    if sample:
        df = df.sample(sample)

    CONTENT_COL = "content"
    df[CONTENT_COL] = df[CONTENT_COL].replace(['\n','\r'],' ', regex=True)

    if dataset == "hp":
        DESCRIPTION_COL = "short_description"
        HEADLINE_COL = "headline"

        df[news_all_data] = df[CONTENT_COL] + " " + df[DESCRIPTION_COL] + " " + df[HEADLINE_COL]

        print("merged columns for hp")
    
    elif dataset == "all_the_news":
        HEADLINE_COL = "title"

        df[news_all_data] = df[CONTENT_COL] + " " + df[HEADLINE_COL]

        print("merged columns for all_the_news")

    # Drop NA's and drop columns where there's only the string "Invalid file"
    df = df.dropna(subset=[news_all_data]).reset_index()
    df = df[df['content'] != "Invalid file"].reset_index(drop=True)
    
    # This creates a column with minor preprocessing wether we need it or not.
    # If all_true is set to all_true=False the news_all_data column in the returned df
    # is going to be equal to this one
    df['minor_preprocessing'] = df[news_all_data]

    print("minor_preprocessing done")

    #import ipdb; ipdb.set_trace()
    # 1.1 lowercase "news + headline" column
    if lower_case or all_true: 
        df[news_all_data] = df[news_all_data].str.lower()

    print("lowercase")

    # 1.2 Create number of "news + headline" decimals column (we need to find a way to use this in the future)
    df['nrs_count'] = df[news_all_data].str.count('[+-]?([0-9]*[.])?[0-9]+')
    df['nrs_count'] = df['nrs_count'].fillna(0)
    df['nrs_count'] = df['nrs_count'].astype(float).astype(int)

    print("nrs count")

    # 1.3 remove digits from news_all_data column
    if no_digits or all_true:
        #import ipdb; ipdb.set_trace()
        df[news_all_data] = df[news_all_data].str.replace('[+-]?([0-9]*[.])?[0-9]+', '', regex=True)
        
        # outra maneira
        # apply(lambda x: ''.join(word for word in x if not word.isdigit()))
        
        print("remove_digits")

    # 1.4 Create support data columns for emotions (we need to find a way to use this column in the future)
    df['questions'] = df[news_all_data].str.count('\?')
    df['exclamations'] = df[news_all_data].str.count('\!')
    df['irony'] = df[news_all_data].map(lambda x: len(re.findall('\?!|\!\?',x)))

    print("emotions")

    # 1.5 Remove punctuation
    if rem_punct or all_true:
        real_string_punctuation = string.punctuation + "—" + '”' + "’" + "‘" + "…" + '“' + '´' + "`" + "«" + "»"
        # import ipdb; ipdb.set_trace()
        df[news_all_data] = df[news_all_data].apply(lambda x: ''\
                                            .join(word for word in x if word not in real_string_punctuation))

        print("punctuation")

    # 1.6 Remove emojis
    if de_emojify or all_true:
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
        
        df[news_all_data] = df["news_all_data"].apply(lambda x: deEmojify(x))

    print("de_emojify")

    if tokenize or all_true:
    # 1.7 Tokenize
        df[news_all_data] = df[news_all_data].apply(lambda x: word_tokenize(x))

    print("tokenize")

    # 1.8 Remove stopwords
    if rem_stopwords or all_true:
        stop_words = set(stopwords.words('english'))
        df[news_all_data] = df[news_all_data]\
                                .apply(lambda x: [word for word in x if not word in stop_words])

        print("stopwords")

    # 1.9a Stemming (optional)
    if stem_not_lematize:
        stemmer = SnowballStemmer(language='english')
        
        df[news_all_data] = df[news_all_data]\
                                    .apply(lambda x: [stemmer.stem(word) for word in x])

        print("stemming")

    # 1.9b Lematizing with POS tags in english (optional)
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

   # 1.10 Adding vocab richness column (we need to find a way to use this column in the future)
    def vocab_richness(text):
        tokens = word_tokenize(text)
        total_length = len(tokens)
        unique_words = set(tokens)
        unique_word_length = len(unique_words)
       
        if total_length > 0:
            return unique_word_length / total_length
        return 0
    
    if lematize or stem_not_lematize or rem_stopwords or all_true:
        df[news_all_data] = df[news_all_data].map(lambda x: ' '.join(x))
    df['vocab richness'] = df[news_all_data].apply(lambda x: vocab_richness(x))

    print("vocab_richness")

    # 1.11 Add column label with translated categories
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

    df['label'] = df.category.map(lambda x: my_dict[x])
  
    return df

if __name__ == "__main__":
    from big_picture.get_merged_data import get_data

    df1 =  get_data(REL_PATH_INPUT)

    df2 = pre_process(df1,
                    dataset="all_the_news",
                    sample=None,
                    all_true=True, 
                    lower_case=False, 
                    no_digits=False,
                    rem_punct=False,
                    de_emojify=False,
                    tokenize=False,
                    rem_stopwords=False,
                    stem_not_lematize=False,
                    lematize=False)

    #for i in range(10):
    #    print(df2.news_all_data.iloc[i])

    df2.to_csv(r'./data/teste_all_the_news.csv', index = False, header=True)
