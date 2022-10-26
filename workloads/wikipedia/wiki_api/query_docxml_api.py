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
def query_doc_revisions(doc_xml_dir="/data/devangjhabakh/wikipedia/wikipedia/doc_xml/", titles_file="/data/devangjhabakh/wikipedia/wikipedia/top_titles_1k.csv", start_time="2022-10-01T02:45:57Z", end_time="2022-08-01T00:00:00Z"):
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
    completed_titles = set(os.listdir("/data/devangjhabakh/wikipedia/wikipedia/doc_xml/"))
    # titles = [title.replace("/", "-").replace(" ", "-") for title in titles]
    # all_titles = set(titles)
    # print(len(all_titles))
    # print(len(completed_titles))
    # print(len(all_titles.difference(completed_titles)))
    # to_do = all_titles.difference(completed_titles)
    dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
    dt_end = datetime.fromisoformat(end_time.replace("Z", ""))
    try:
        for title in tqdm(titles):
            if title.replace("/", "-").replace(" ", "-") in completed_titles:
                print("skipping title!")
                print(title)
                continue
            print(title)

            dt_start = datetime.fromisoformat(start_time.replace("Z", ""))
            dt_end = datetime.fromisoformat(end_time.replace("Z", ""))

            while dt_end < dt_start: 

                PARAMS["pages"] = title
                PARAMS["offset"] = dt_start.isoformat() + "Z"

                title = title.replace(" ", "-")
                title = title.replace("/", "-")

                # make request
                try:
                    R = S.post(url=URL, params=PARAMS)
                except Exception as e:
                    f = open("log_file.txt", "a")
                    print(str(e))
                    f.write(str(e))

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
                try:
                    page_child = [c for c in contents if c.tag == 'page'][0]
                except Exception as e:
                    # no more pages to load, so we're done and have run out
                    break
                title = page_child[0].text
                ns = page_child[1].text
                pg_id = page_child[2].text
                
                # find the start of revisions
                rev_start = len(page_child) - 1
                for i, page in enumerate(page_child):
                    if page.tag == "revision":
                        rev_start = i
                        break

                # go through all the revisions
                for i in (range(rev_start, len(page_child))):
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
                    filename = os.path.join(data_dir, title.replace("/", "-").replace(" ", "-"))
                    file_obj = open(filename, "ab")
                    file_obj.write(curr_bytes)

                    # update timestamp
                    if dt < dt_start:
                        dt_start = dt
                    else:
                        break

                print("wrote to " + str(title))
    except Exception as e:
        f = open("log_file.txt", "a")
        print(str(e))
        f.write(str(e))

if __name__ == "__main__":
    query_doc_revisions()
