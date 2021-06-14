# imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.nn import softmax
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras import Sequential, layers # may be necessary down the line

news_all_data = "news_all_data" # gotta find a better way to do this...

def distilbert(df, max_length=500):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    import ipdb; ipdb.set_trace()
    texts = list(df[news_all_data])

    encoded_input = tokenizer(texts, 
                          return_tensors='tf',
                          padding=True,
                          max_length=max_length,
                          truncation=True)
    
    output = model(encoded_input)

    my_array = softmax(output.logits).numpy()

    df = pd.DataFrame(my_array, columns = ['Negative','Positive'])

    df['Result'] = df['Positive'] - df['Negative']

    # Optional Scalling (we may find out that news are not mostly negatively biased)
    scaler = MinMaxScaler(feature_range=(-1, 1)) # Instanciate StandarScaler
    scaler.fit(df[['Result']]) # Fit scaler to data
    df['Scaled_Result'] = scaler.transform(df[['Result']]) # Use scaler to transform data

    return df

if __name__ == "__main__":
    from big_picture.get_merged_data import get_data

    REL_PATH_INPUT = "../raw_data/data_12k/"
    CONTENT_COL = "content"
    DESCRIPTION_COL = "short_description"
    HEADLINE_COL = "headline"

    df =  get_data(REL_PATH_INPUT)
    df = df.sample(15)
    df[CONTENT_COL] = df[CONTENT_COL].replace('\n',' ', regex=True)

    df[news_all_data] = df[CONTENT_COL] + " " + df[DESCRIPTION_COL] + " " + df[HEADLINE_COL]
    df = df.dropna(subset=[news_all_data]).reset_index()
    df = df[df[news_all_data] != "Invalid file"].reset_index(drop=True)

    print(distilbert(df, max_length=500))