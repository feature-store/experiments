#!/usr/bin/python3
import requests
from datetime import datetime
import os
import json
 
def query_recent_changes(start_time="2022-10-01T02:45:57Z", end_time="2022-08-01T00:00:00Z", changes_dir="/data/jeffcheng1234/wikipedia/wikipedia/recentchanges"):
    """
    Script to query the wikipedia API for recent edit IDS
    """

    ## SET THESE
    #data_dir = "revisions/"
    data_dir = "/data/jeffcheng1234/wikipedia/wikipedia/recentchanges"

    # setup API 
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    #"https://en.wikipedia.org/w/api.php?action=query&list=recentchanges&format=json&rclimit=500&rcstart=2021-08-29T16:00:04Z"


    PARAMS = {
        "action": "query",
        "rcnamespace": "0", #only query wikipedia namespace https://en.wikipedia.org/wiki/Wikipedia:Namespace
        "list": "recentchanges", 
        "rcprop": "user|userid|comment|flags|timestamp|title|ids|sizes|tags",
        "rcshow": "!minor|!bot",
        "rctype": "edit",
        "rcslot": "main", # main slot?
        "rclimit": 500,
        "rcstart": start_time, # put the last timestap here to get more
        "format": "json"
    }
    
    orig_start_time = start_time
    dt_start = datetime.strptime(start_time.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    dt_end = datetime.strptime(end_time.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")

    revisions = []
    while dt_end < dt_start: 

        # update start time
        start_time = dt_start.isoformat() + "Z"
        PARAMS["rcstart"] = start_time

        # make request
        print(PARAMS)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        changes = DATA["query"]["recentchanges"]
        #print(DATA)
        print(len(changes))
        if len(changes) == 0:
            dt_start = datetime.fromtimestamp(datetime.timestamp(dt_start) - 700)
        for change in changes:

            # note - "revid" should equal "parentid" for some later diff
            dt = datetime.strptime(change["timestamp"].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")

            # add to list of revisions
            revisions.append(change)

            # update startime
            #print(dt)
            if dt < dt_start: 
                #print("update", dt_start.isoformat(), dt.isoformat())
                dt_start = dt

        # write output
        if len(revisions) % 10000 == 0: 
            filename = os.path.join(data_dir, f"revision_stream_{start_time}_{orig_start_time}.json")
            print(filename)
            open(filename, "w").write(json.dumps(revisions))
            orig_start_time = dt_start.isoformat() + "Z"

            revisions = []

if __name__ == "__main__":
    query_recent_changes()