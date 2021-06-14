# Imports
import pandas as pd
import requests
import json
from datetime import date
from dateutil import relativedelta
import random

# Scraping from API: https://newsapi.org/v2/

def get_random_everything_news_api(api, nr_requests):

# Getting a list of random dates within a range of 1 month from "today"
    def get_dates_list_one_month_range():
        lastmonth = date.today() + relativedelta.relativedelta(months=-1)
        delta_days = (date.today() - lastmonth).days
        dates_list = []

        for i in range(1,delta_days+1):
            dates_list.append((lastmonth + relativedelta.relativedelta(days=i)).strftime("%Y-%m-%d"))
        
        return dates_list

    # Getting a list of 20 random sources out of 81 possible english language sources

    def random_sources():
        get_sources = open('../raw_data/news_api/sources.json',) 
        data = json.load(get_sources)
        get_sources.close()

        sources_df = pd.DataFrame(data["sources"])
        sources_df = sources_df[sources_df["language"] == "en"]
        sources_list = sources_df.id.to_list()
        # the maximum number of sources is 20 (there are 81 language sources in total)
        sources_list = random.sample(sources_list, 20)
        random_sources = ",".join(sources_list)
        
        return random_sources

    lastmonth = date.today() + relativedelta.relativedelta(months=-1)
    delta_days = (date.today() - lastmonth).days

    files_list = []
    date_iterator = get_dates_list_one_month_range()

    # the maximum number of requests per day is 100
    # the maximum number of news articles we can get per page is 100
    # Also, if we request 100 articles per page we can only retrieve one page max per request.

    # Getting a request response from News API
    for i in range(nr_requests):
        sources = random_sources()

        if i >= delta_days:
            date_iterator = random.sample(get_dates_list_one_month_range(),1)[0]
        else:
            date_iterator = get_dates_list_one_month_range()[i]

        url = 'https://newsapi.org/v2/'
        url_params = 'everything' #everything or top headlines
        params = { 'sources': sources,
                    'pageSize': 100,
                    'from': date_iterator,
                    'to': date_iterator,
                    'language':'en',
                    'apiKey': api}

        response = requests.get(url + url_params, params=params)

        # Putting the response into a pandas df
        file = response.json()
        files_list.append(file["articles"])

        elems = []
        for file in files_list:
            for elem in file:
                elems.append(elem)
        news_api_df = pd.DataFrame(elems)
        news_api_df = news_api_df.drop_duplicates(subset=['content'])
        news_api_df['content'] = news_api_df['content'].replace(['\n','\r'],' ', regex=True)

    return news_api_df

if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()

    api = "940f51da7bc0439b8cf24417ae0bd5e4" # API jmartins
    nr_requests = 2

    print(get_random_everything_news_api(api, nr_requests))
