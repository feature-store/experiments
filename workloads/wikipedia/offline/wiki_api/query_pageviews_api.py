#!/usr/bin/python3
import requests
from datetime import datetime
import os
import json
import pandas as pd
import tqdm

def get_url(title, start_time, end_time):
    title = title.replace("?", "%3F")
    title = title.replace(" ", "_")
    URL = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{end_time}/{start_time}"
    return URL

#gets pageview across some time interval for a single article
def get_article_pageview(session, title, start_time, end_time):
    row = {}
    row["title"] = title

    URL = get_url(title, start_time, end_time)
    PARAMS = {
        "format": "json"
    }
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    # print(URL)
    R = session.get(url=URL,params=PARAMS, headers=headers)
    DATA = R.json()
    print(DATA)
    #list of views from start_time to end_time
    pageview_results = DATA["items"]
    for pageview_result in pageview_results:
        day = pageview_result["timestamp"] #YYYYMMDD
        row[day] = pageview_result["views"]
    
    print(f"number of days for {title} is {len(row) - 1}")

    return row

def query_pageviews(raw_pageview_file,titles_file,start_time="2022-10-01T02:45:57Z", end_time="2022-08-01T00:00:00Z"):
    """
    Script to query the wikipedia API for articles' pageview counts within a given interval 
    """

    #parse time to YYYYMMDD
    start_time = start_time[:-10].replace("-", "")
    end_time = end_time[:-10].replace("-", "")
    
    # setup API 
    S = requests.Session()

    titles = pd.read_csv(titles_file)
    raw_pageview = []
    for title in tqdm.tqdm(titles["title"]):
        try:
            row_for_title = get_article_pageview(S, title, start_time, end_time)
            raw_pageview.append(row_for_title)
        except:
            print(f"exception occurred, can't get pageview for article {title}, url: {get_url(title, start_time, end_time)}")

    raw_pageview = pd.DataFrame.from_dict(raw_pageview).fillna(0)
    print(f"shape of raw_pageview: {raw_pageview.shape}")

    raw_pageview.to_csv(raw_pageview_file, index=False)


if __name__ == "__main__":
    S = requests.Session()
    start_time="20220801"
    end_time="20221001"
    title_pageview = get_article_pageview(S, "Deaths in 2021", end_time, start_time)
    print(title_pageview)