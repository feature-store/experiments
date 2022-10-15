#!/usr/bin/python3
import requests
from datetime import datetime
import os
import json
import pandas as pd

"""
Script to query all the document content version in a given time period
"""

# SET THESE
titles_df = pd.read_csv("titles.csv")
titles = titles_df["title"].tolist()
data_dir = "/data/devangjhabakh/wikipedia/wikipedia/revisions/"
print("num titles", len(titles))

 
# make sure this matches the other script
start_time = "2022-10-01T02:45:57Z"
end_time =  "2022-08-01T00:00:00Z" # timestamp to stop at 
# setup API 
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "prop": "revisions",
    "titles": "", # set later
    "rvprop": "ids|timestamp|user|comment|content|flags|size|contentmodel|slotsize",
    "rvslots": "main",
    "rvlimit": 500,
    "formatversion": "2",
    #"oldid": "1040989683", 
    #"diff": "1040996201",
    #"rvstart": "2021-07-20T14:13:29Z", # put the last timestap here to get more
    "format": "json"
}

for title in titles: 
    print(title)

    dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
    dt_end = datetime.fromisoformat(end_time.replace("Z", ""))

    while dt_end < dt_start: 

        PARAMS["titles"] = title
        PARAMS["rvstart"] = dt_start.isoformat() + "Z"

        # make request
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        PAGES = DATA["query"]["pages"]
        for page in PAGES:
            print(page["pageid"], page["title"], page["ns"])
            for revision in page["revisions"]:

                # note - "revid" should equal "parentid" for some later diff
                print(revision["revid"], revision["timestamp"], page["pageid"])

                dt = datetime.fromisoformat(revision["timestamp"].replace("Z", ""))
                filename = title.replace(" ", "-") + "_" + revision["timestamp"] + ".json"

                # dump versions info 
                open(os.path.join(data_dir, filename), "w").write(json.dumps(revision, indent=2))

                # update start time for next request
                if dt < dt_start: 
                    dt_start = dt
