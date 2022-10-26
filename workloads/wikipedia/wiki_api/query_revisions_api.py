#!/usr/bin/python3
import requests
from datetime import datetime
import os
import xml.etree.ElementTree as ET
import pandas as pd

def query_revisions_api(titles_file, raw_doc_dir, start_time="2022-10-01T02:45:57Z", end_time ="2022-08-01T00:00:00Z"):
    """
    Script to query all the document content version in a given time period
    """

    # SET THESE
    titles_df = pd.read_csv(titles_file)
    titles = titles_df["title"].tolist()
    data_dir = raw_doc_dir
    print("num titles", len(titles))

    # setup API 
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        # "prop": "revisions",
        "titles": "", # set later
        "rvprop": "ids|timestamp|user|comment|content|flags|size|contentmodel|slotsize",
        "rvslots": "main",
        "rvlimit": 500,
        # "formatversion": "2",
        #"oldid": "1040989683",
        #"diff": "1040996201",
        #"rvstart": "2021-07-20T14:13:29Z", # put the last timestap here to get more
        "format": "xml"
    }

    for title in titles: 
        print(title)

        dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
        dt_end = datetime.fromisoformat(end_time.replace("Z", ""))

        while dt_end < dt_start: 

            PARAMS["titles"] = "2021 British Open"
            PARAMS["rvstart"] = dt_start.isoformat() + "Z"

            # make request
            R = S.get(url=URL, params=PARAMS)

            try:
                contents = ET.fromstring(R.content)
                for child in contents.findall('./'):
                    print(child, child.attrib)
                    print([c for c in child])
            except Exception as e:
                print(str(e)[:100])

            break
            # PAGES = DATA["query"]["pages"]
            # for page in PAGES:
            #     print(page["pageid"], page["title"], page["ns"])

            #         # note - "revid" should equal "parentid" for some later diff
            #         print(revision["revid"], revision["timestamp"], page["pageid"])

            #         dt = datetime.fromisoformat(revision["timestamp"].replace("Z", ""))
            #         filename = title.replace(" ", "-") + "_" + revision["timestamp"] + ".json"

            #         # dump versions info 
            #         open(os.path.join(data_dir, filename), "w").write(json.dumps(revision, indent=2))

            #         # update start time for next request
            #         if dt < dt_start: 
            #             dt_start = dt
        raise Exception("aa")

query_revisions_api("/data/devangjhabakh/wikipedia/wikipedia/top_titles.csv", ".")