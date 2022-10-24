#!/usr/bin/python3
import requests
from datetime import datetime
import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree
from tqdm import tqdm
 
# def query_doc_versions(start_time="2022-10-01T02:45:57Z", end_time="2022-08-01T00:00:00Z", changes_dir="/data/devangjhabakh/wikipedia/wikipedia/recentchanges"):
def query_doc_versions(doc_xml_dir="/data/devangjhabakh/wikipedia/wikipedia/doc_xml/", titles_file="/data/devangjhabakh/wikipedia/wikipedia/top_titles.csv", start_time="2022-10-01T02:45:57Z", end_time="2022-08-01T00:00:00Z"):
    """
    Script to query the wikipedia API for recent edit IDS
    """

    top_titles = pd.read_csv(titles_file)
    titles = list(set(top_titles["title"].tolist()))

    ## SET THESE
    #data_dir = "revisions/"
    data_dir = doc_xml_dir

    # setup API 
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/index.php"


    PARAMS = {
        "title": "Special:Export",
        "limit": 1000,
        "dir": "desc"
    }
    
    orig_start_time = start_time
    dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
    dt_end = datetime.fromisoformat(end_time.replace("Z", ""))

    for title in tqdm(titles): 
        print(title)

        dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
        dt_end = datetime.fromisoformat(end_time.replace("Z", ""))

        while dt_end < dt_start: 

            PARAMS["pages"] = title
            PARAMS["offset"] = dt_start.isoformat() + "Z"

            # make request
            R = S.post(url=URL, params=PARAMS)

            try:
                contents = etree.fromstring(R.content)
                for elem in contents.getiterator():
                    # Skip comments and processing instructions,
                    # because they do not have names
                    if not (
                        isinstance(elem, etree._Comment)
                        or isinstance(elem, etree._ProcessingInstruction)
                    ):
                        # Remove a namespace URI in the element's name
                        elem.tag = etree.QName(elem).localname

                # Remove unused namespace declarations
                etree.cleanup_namespaces(contents)
                page_child = [c for c in contents if c.tag == 'page'][0]
                title = page_child[0].text
                ns = page_child[1].text
                pg_id = page_child[2].text

                # go through all the revisions
                for i in (range(3, len(page_child))):
                    revision = page_child[i]

                    curr_timestamp = [c for c in revision if c.tag == 'timestamp'][0].text
                    dt = datetime.fromisoformat(curr_timestamp.replace("Z", ""))

                    ## add beginning of xml
                    curr_bytes = bytes('<?xml version="1.0" ?>\n<page>\n', 'utf-8') \
                                + bytes('<id>' + pg_id + '</id>\n', 'utf-8') \
                                + bytes('<title>' + title + '</title>\n', 'utf-8') \
                                + bytes('<ns>' + ns + '</ns>\n', 'utf-8') \
                                + etree.tostring(revision, pretty_print=True) \
                                + bytes('</page>\n', 'utf-8')

                    ## write xml to appropriate file
                    filename = os.path.join(data_dir, title.replace(" ", "-"))
                    file_obj = open(filename, "ab")
                    file_obj.write(curr_bytes)

                    print("wrote to " + str(filename))

                    # update timestamp
                    if dt < dt_start:
                        dt_start = dt

            except Exception as e:
                print(str(e))

if __name__ == "__main__":
    query_doc_versions()