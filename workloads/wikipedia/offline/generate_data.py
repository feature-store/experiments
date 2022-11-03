import os
import time
import pickle
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import subprocess

import configparser
import argparse

import pandas as pd
import numpy as np

from multiprocessing import Pool
from bs4 import BeautifulSoup
from scipy.stats.mstats import winsorize

# import wandb

# from concurrent.futures import ProcessPoolExecutor

# from generate diffs file (originally from DPR repo... sorry kevin)
from preprocessing.generate_diffs import generate_sentence_level_diffs
from preprocessing.embedding import generate_embeddings

from log_data import log_files, log_pageview, log_simulation, log_questions


def query_recentchanges(start_time, end_time, revision_file):
    pass


def query_doc_versions(titles_file, start_time, end_time, raw_doc_dir):
    # TODO: query doc versions
    titles_df = pd.read_csv(titles_file)
    titles = list(set(titles_df.index.tolist()))
    pass


def get_recent_changes(revisions_dir, changes_file):
    changes = []
    revids = set([])
    files = os.listdir(revisions_dir)
    for i in range(len(files)):
        f = files[i]
        f_changes = json.loads(open(os.path.join(revisions_dir, f), "r").read())

        for change in f_changes:
            if change["revid"] in revids:
                continue

            changes.append(change)
            revids.add(change["revid"])

        # if i % 100 == 0:
        #    print(f"Read {i}/{len(files)}, changes so far: {len(changes)}")

    print(len(files))
    # create dataframe
    changes_df = pd.DataFrame(changes)

    # create time index
    print(changes_df.head(10))
    changes_df["datetime"] = pd.to_datetime(changes_df["timestamp"])
    changes_df = changes_df.set_index("datetime").sort_index()

    # save to CSV file
    changes_df.to_csv(changes_file)

    return changes_df


def get_titles(changes_file, titles_file, n=200):
    changes_df = pd.read_csv(changes_file)
    title_ids = set(changes_df[["title", "pageid"]].apply(tuple, axis=1).tolist())

    counts = changes_df.title.value_counts().to_frame()
    top_titles = counts.nlargest(n, "title")
    print(top_titles)
    top_titles.columns = ["count"]
    top_titles["title"] = top_titles.index
    top_titles.to_csv(titles_file)
    return top_titles


def get_edits(edits_file, changes_file, titles_file):
    changes_df = pd.read_csv(changes_file)
    titles_df = get_titles(changes_file, titles_file)
    titles = list(set(titles_df.index.tolist()))
    edits_df = changes_df[changes_df.title.apply(lambda x: x in titles)]

    # assign timestamps
    edits_df["ts_min"] = (
        pd.to_datetime(edits_df["datetime"])
        .astype(np.int64)
        .apply(assign_timestamps_min)
    )

    # write CSV
    edits_df.to_csv(edits_file)
    return edits_df


def get_questions(raw_questions_file, questions_file):
    questions_df = pd.read_csv(raw_questions_file, sep="\t")
    questions_df.columns = [
        "question",
        "answer",
        "doc_id",
        "datetime",
        "revid",
        "oldrevid",
    ]

    # assign timestamps
    qdf["ts_min"] = (
        pd.to_datetime(questions_df["datetime"])
        .astype(np.int64)
        .apply(assign_timestamps_min)
    )

    # write CSV
    questions_df.to_csv(questions_file)
    return questions_df

def get_raw_pageviews(raw_pageview_file, titles_file, start_time, end_time):
    from wiki_api.query_pageviews_api import query_pageviews
    query_pageviews(raw_pageview_file, titles_file, start_time, end_time) 

def get_pageviews(raw_pageview_file, pageview_file, edits_file, timestamp_weights_file): 

    edits_df = pd.read_csv(edits_file)
    pageview_df = pd.read_csv(raw_pageview_file)

    # map title -> id
    title_to_id = edits_df.set_index("title")["pageid"].to_dict()
    open("title_to_id.json", "w").write(json.dumps(title_to_id))

    # get edit counts for each page
    edit_counts_df = edits_df.groupby("title").size()
    pageview_df['edit_count'] = list(edit_counts_df[pageview_df["title"]])
    cols = pageview_df.columns.tolist()
    pageview_df = pageview_df[cols[:1] + cols[-1:] + cols[1:-1]]

    # scale down the outliers with its square root and clip the 10th and 60th percentile values 
    # with winsorization to ensure some visits without skewing the distribution too much
    for c in pageview_df.columns.tolist()[2:]:
        pageview_df[c] = (pageview_df[c] ** 0.5).astype(int)
        pageview_df[c] = winsorize(pageview_df[c], limits=[0.2, 0.05])

    # calculate page weights
    total_views = pageview_df.iloc[:, 2:].sum(axis=1).sum()
    weights = pageview_df.iloc[:, 2:].sum(axis=1) / total_views 
    pageview_df['weights'] = weights
    pageview_df['doc_id'] = pageview_df['title'].apply(lambda x: title_to_id[x])
        
    # pageview_df = pageview_df.drop(columns=["Unnamed: 0"])
    pageview_df 

    # page weights per timestamp
    ts_to_weights = {}
    dates = pageview_df.columns[2:-2]
    for date in dates: 
        print(date)
        dt = datetime.strptime(date[:-2], '%Y%m%d')
        ts = dt.timestamp() * 1000000000
        ts_min = assign_timestamps_min(ts)
        view_counts = pageview_df[date].tolist()
        id_to_count  = pageview_df.set_index("doc_id")[date].to_dict()
        ts_to_weights[ts_min] = id_to_count
    open(timestamp_weights_file, "w").write(json.dumps(ts_to_weights))
    print("Generated ts weights file", timestamp_weights_file)
    return pageview_df


# create diff JSON file from valid list of revision pairs, doc pkl
def create_diff_json(doc_pkl, rev_pairs, diff_dir):

    # load data for file
    data = pickle.loads(open(doc_pkl, "rb").read())
    title = os.path.basename(doc_pkl).replace(".pkl", "")

    for i in range(len(data)):
        orig_doc = data[i]

        for j in range(0, len(data), 1):
            new_doc = data[j]

            rev_pair = orig_doc["id"] + "_" + new_doc["id"]

            if rev_pair not in rev_pairs:
                continue

            diff_file = os.path.join(diff_dir, rev_pair + ".json")
            if os.path.exists(diff_file):
                # skip
                continue

            edits = {orig_doc["title"]: new_doc}
            try:
                all_diffs = generate_sentence_level_diffs([orig_doc], edits)
            except Exception as e:
                print(e)
                raise ValueError(f"Failed to parse diffs {rev_pair}")
            diff = {
                "title": orig_doc["title"],
                "timestamp": rev_pairs[rev_pair],
                "orig_id": orig_doc["id"],
                "new_id": new_doc["id"],
                "diffs": all_diffs,
            }
            open(diff_file, "w").write(json.dumps(diff, indent=2))


def generate_diffs_helper(filename, diff_dir, rev_pair, timestamp):

    data = pickle.loads(open(filename, "rb").read())

    for i in range(len(data)):
        for j in range(len(data)):
            orig_doc = data[i]
            new_doc = data[j]

            if new_doc["id"] + "_" + orig_doc["id"] != rev_pair:
                continue

            # parse diffs
            diff_file = os.path.join(diff_dir, rev_pair + ".json")

            if os.path.exists(diff_file):
                continue

            edits = {orig_doc["title"]: new_doc}
            st = time.time()
            all_diffs, has_diff = generate_sentence_level_diffs([orig_doc], edits)
            # print("runtime", time.time() - st)
            diff = {
                "title": orig_doc["title"],
                "timestamp": timestamp,
                "orig_id": orig_doc["id"],
                "new_id": new_doc["id"],
                "diffs": all_diffs,
            }
            if has_diff:
                diff = {
                    "title": orig_doc["title"],
                    "timestamp": timestamp,
                    "orig_id": orig_doc["id"],
                    "new_id": new_doc["id"],
                    "diffs": all_diffs,
                }
            else:
                diff = {
                    "title": orig_doc["title"],
                    "timestamp": timestamp,
                    "orig_id": orig_doc["id"],
                    "new_id": new_doc["id"],
                    "diffs": [],
                }
            # TODO: write to tmp file first (make sure we dont have messed up files)
            open(diff_file, "w").write(json.dumps(diff, indent=2))
            return


def generate_diffs(
    edits_file, titles_file, parsed_doc_dir, diff_dir, revision_file, workers=32
):

    # make sure title is in titles df
    titles_df = pd.read_csv(titles_file)
    titles = list(set(titles_df.title.tolist()))

    # print(titles)

    # filter out revision pairs not in edits_file
    edits_df = pd.read_csv(edits_file)
    title_to_rev_pairs = defaultdict(dict)
    for index, row in edits_df.iterrows():
        if row["title"] not in titles:
            continue  # skip if not top title

        # map title -> (revid, old_revid) -> timestamp of revision
        rev_pair = str(row["revid"]) + "_" + str(row["old_revid"])
        title_to_rev_pairs[row["title"]][rev_pair] = row["timestamp"]

    open(revision_file, "w").write(json.dumps(title_to_rev_pairs))

    num_keys = len(title_to_rev_pairs.keys())
    # print(
    #    f"Proceessing revisions for {num_keys} titles, writing to {diff_dir}"
    # )

    inputs = []
    for title in tqdm(titles):
        filename = os.path.join(parsed_doc_dir, f"{title}.pkl")
        if not os.path.exists(filename):
            print("missing", filename)
            continue

        for rev_pair in title_to_rev_pairs[title].keys():
            if os.path.exists(os.path.join(diff_dir, rev_pair + ".json")):
                continue
            inputs.append(
                (filename, diff_dir, rev_pair, title_to_rev_pairs[title][rev_pair])
            )

    print("processing revids", len(inputs), diff_dir)
    chunk_size = 100000
    for i in range(0, len(inputs), chunk_size):
        p = Pool(128)
        print("created pool", i, i + chunk_size, len(inputs))
        p.starmap(generate_diffs_helper, inputs[i : i + chunk_size])
        p.close()

    return

    # diff remaining
    inputs = [
        (
            os.path.join(parsed_doc_dir, f"{title}.pkl"),
            title_to_rev_pairs[title],
            diff_dir,
        )
        for title in titles
    ]
    p = Pool(workers)
    p.starmap(create_diff_json, inputs)
    p.close()


# convert wikipedia dump into single pkl file per title
def dump_to_pickle_title(top_folder, target_dir, title):
    total = 0
    docs = []
    for folder in os.listdir(top_folder):
        for file in os.listdir(os.path.join(top_folder, folder)):

            filename = os.path.join(top_folder, folder, file)
            data = open(filename, "r").read()
            soup = BeautifulSoup(data, "html.parser")

            for doc in soup.find_all("doc"):
                id = doc.get("id")
                title = doc.get("title")
                url = doc.get("url")
                text = doc.get_text()
                docs.append({"id": id, "url": url, "title": title, "text": text})
        total += len(docs)
    pickle.dump(docs, open(os.path.join(target_dir, title + ".pkl"), "wb"))
    return os.path.join(target_dir, title + ".pkl")


# call wikiextractor library on XML
def extract(title, raw_doc_dir, parsed_tmp_dir, parsed_doc_dir):
    f = f"{raw_doc_dir}/{title}"
    bashCommand = f"wikiextractor {f} -o {parsed_tmp_dir}/tmp_parsed{title}"

    print(bashCommand)

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    pkl_file = dump_to_pickle_title(
        f"{parsed_tmp_dir}/tmp_parsed{title}", parsed_doc_dir, title
    )


def parse_docs(raw_doc_dir, parsed_tmp_dir, parsed_doc_dir, workers=32):
    # parse documents from raw XML

    # extract individual doc
    files = os.listdir(raw_doc_dir)
    # TODO: add assert to make sure titles correspond to filenames
    files = [
        (f, raw_doc_dir, parsed_tmp_dir, parsed_doc_dir)
        for f in files
        if not os.path.isdir(f)
    ]
    
    # create pool and run
    p = Pool(workers)
    p.starmap(extract, files)
    p.close()


# assign timesteps
def assign_timestamps_min(ts):
    # take in unix timestamp - covert to integer
    start_ts = 1628131044000000000  # don't change
    delta = ts - start_ts
    if delta < 0:
        return None

    return int(delta / (60 * 1000000000))


def generate_simulation_data(
    questions_file,
    edits_file,
    diff_dir,
    init_data_file,
    stream_edits_file,
    stream_questions_file,
):
    edits_df = pd.read_csv(edits_file)
    questions_df = pd.read_csv(questions_file)

    # lists for questions/edits at each timestep
    questions = []
    edits = []

    # initialization data for embeddings/passages
    init_data = {}

    # timestamp to stop
    max_ts = int(questions_df.ts_min.max())

    # loop through timestamps
    for ts in range(max_ts + 1):

        ts_edits = defaultdict(list)
        ts_queries = defaultdict(list)
        for index, row in edits_df[edits_df["ts_min"] == ts].iterrows():
            filename = str(row["revid"]) + "_" + str(row["old_revid"]) + ".json"
            key = row["pageid"]

            # make sure file is OK
            file_path = os.path.join(diff_dir, filename)
            if os.path.exists(file_path):
                try:
                    data = json.load(open(file_path))
                    if len(data["diffs"]) == 0:
                        continue
                    diffs = data["diffs"][0]
                except Exception as e:
                    print(file_path)
                    print(e)
                    continue
                diff_types = [
                    d["diff_type"] for d in diffs if d["diff_type"] is not None
                ]
                if len(diff_types) == 0:
                    print(f"Invalid file {filename}")
                    continue
                assert str(data["orig_id"]) == str(
                    row["old_revid"]
                ), f"Invalid id {filename}, id {data['orig_id']} row {row['revid']}"

                # get length of passage

                if key not in init_data:
                    diffs = data["diffs"][0]
                    init_data[key] = {
                        "revid": data["orig_id"],
                        "sents": [d["sent_a"] for d in diffs],
                        "file": filename,
                        "ts_min": row["ts_min"],
                    }
                ts_edits[key].append(filename)

            else:
                # print("missing", file_path)
                continue

        for index, row in questions_df[questions_df["ts_min"] == ts].iterrows():
            key = row["doc_id"]
            ts_queries[key].append(
                {
                    "question": row["question"],
                    "doc_id": key,
                    "answer": row["answer"],
                    "datetime": row["datetime"],
                    "ts_min": row["ts_min"],
                    "revid": row["revid"],
                    "old_revid": row["oldrevid"],
                }
            )

        edits.append(ts_edits)
        questions.append(ts_queries)

        if ts % 1000 == 0:
            unique_files = set([])
            for e in edits:
                for files in e.values():
                    for f in files:
                        unique_files.add(f)
            print(f"Num edits ts {ts}/{max_ts+1}: {len(unique_files)}")

    open(stream_edits_file, "w").write(json.dumps(edits))
    open(stream_questions_file, "w").write(json.dumps(questions))
    open(init_data_file, "w").write(json.dumps(init_data))


def search_answer(rev_file, embedding_dir, question):
    # read file and see if answer is contained
    revid = rev_file.replace(".json", "").split("_")[0]
    # assert str(revid) == str(question["revid"]), f"Invalid id {revid}, {question}"
    embedding_filename = os.path.join(embedding_dir, f"{revid}_new.pkl")
    try:
        passages = pickle.load(open(embedding_filename, "rb"))["passages"]
    except Exception as e:
        print(e)
        print("File error", embedding_filename)
        return False

    found_answer = False
    for passage in passages:
        if question["answer"] in passage:
            found_answer = True
    return found_answer

def query_recentchanges_api(start_time, end_time, revisions_dir):
    from wiki_api.query_recentchanges_api import query_recent_changes
    query_recent_changes(start_time, end_time, revisions_dir)

def check_dataset(
    titles_file,
    edits_file,
    init_data_file,
    stream_edits_file,
    stream_questions_file,
    diff_dir,
):
    # TODO: add checks (init data keys match stream keys, questions match keys, etc.)

    # load data
    edits_df = pd.read_csv(edits_file)
    titles_df = get_titles(changes_file, titles_file)
    titles = list(set(titles_df.index.tolist()))
    init_data = json.load(open(init_data_file))
    edits = json.load(open(stream_edits_file))
    questions = json.load(open(stream_questions_file))

    # same length
    assert len(questions) == len(edits)

    for ts in range(len(questions)):
        for doc_id in questions[ts].keys():
            if not doc_id in init_data:
                print("missing doc", doc_id)
                continue
            for question in questions[ts][doc_id]:
                # print(question)
                answer = question["answer"]
                # import pdb; pdb.set_trace()

                # question = questions[ts][doc_id]
                rev_file = (
                    str(question["revid"]) + "_" + str(question["old_revid"]) + ".json"
                )

                if not os.path.exists(os.path.join(diff_dir, rev_file)):
                    print("Still missing diff", rev_file)
                    continue

                # question generated from document edit - assert it was created before
                found = False
                revision_file = None
                found_index = 0
                for i in range(ts):
                    if doc_id in edits[ts - i]:
                        if rev_file in edits[ts - i][doc_id]:
                            found = True
                            revision_file = rev_file
                            found_index = ts - i
                            break
                if not found:
                    # only option is that it was derived from original doc
                    assert str(init_data[doc_id]["revid"]) == str(
                        question["old_revid"]
                    ), f"Missing revision {ts}, {rev_file}, {doc_id}, init version {init_data[doc_id]['revid']}"
                    revision_file = init_data[doc_id]["file"]

                # search for answer in revision file
                found_answer = search_answer(revision_file, embedding_dir, question)
                if not found_answer:
                    print("NOT FOUND", found_answer, revision_file)
                else:
                    print("FOUND", found_answer, revision_file)

                if (
                    question["question"]
                    == "how far is hurricane ida from cuba?????????????????"
                ):
                    print("DEBUG", question)
                    print(rev_file)
                    print("question ts", ts, "edit ts", found_index)
                    for i in range(found_index, ts + 1, 1):
                        if doc_id in edits[i]:
                            print(
                                i,
                                edits[i][doc_id],
                                search_answer(
                                    edits[i][doc_id][-1], embedding_dir, question
                                ),
                            )
                    print(found_answer)

    # docid_to_title = {}
    # for index, row in edits_df.iterrows():
    #    docid_to_title[row["pageid"]] = row["title"]

    # open("docid_to_title.json", "w").write(json.dumps(docid_to_title))

    ## check matching keys
    # last_doc = init_data
    # for i in len(edits):
    #    # TODO: assert that question actually contained in this edit?
    #    continue

    # check each edit is contained

    # check raw edit timestamp is same as query timestamp


if __name__ == "__main__":

    # run = wandb.init(job_type="dataset-creation", project="wiki-workload")

    # configuration file
    config = configparser.ConfigParser()
    config.read("config.yml")

    # argument flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_query_recentchanges", action="store_true", default=False
    )  # query wiki api for recentchanges
    parser.add_argument(
        "--run_query_doc_versions", action="store_true", default=False
    )  # query wiki api for doc versions
    parser.add_argument(
        "--run_recent_changes", action="store_true", default=False
    )  # re-processing api changes data
    parser.add_argument(
        "--run_parse_docs", action="store_true", default=False
    )  # re-parse document versions
    parser.add_argument("--run_get_questions", action="store_true", default=False)
    parser.add_argument("--run_get_raw_pageviews", action="store_true", default=False)
    parser.add_argument("--run_get_pageviews", action="store_true", default=False)
    parser.add_argument(
        "--run_generate_diffs", action="store_true", default=False
    )  # re-process generating diffs
    parser.add_argument(
        "--run_generate_simulation_data", action="store_true", default=False
    )
    parser.add_argument("--run_check_dataset", action="store_true", default=False)
    parser.add_argument("--run_generate_embeddings", action="store_true", default=False)
    args = parser.parse_args()

    # directories
    data_dir = config["directory"]["data_dir"]
    revisions_dir = config["directory"]["revisions_dir"]
    raw_doc_dir = config["directory"]["raw_doc_dir"]
    parsed_doc_dir = config["directory"]["parsed_doc_dir"]
    parsed_tmp_dir = config["directory"]["parsed_tmp_dir"]
    diff_dir = config["directory"]["diff_dir"]
    embedding_dir = config["directory"]["embedding_dir"]

    # intermediate files
    model_file = config["files"]["model_file"]
    changes_file = config["files"]["changes_file"]
    titles_file = config["files"]["titles_file"]
    revisions_file = config["files"]["revisions_file"]
    edits_file = config["files"]["edits_file"]
    raw_questions_file = config["files"]["raw_questions_file"]
    questions_file = config["files"]["questions_file"]
    raw_pageview_file = config["files"]["raw_pageview_file"]
    pageview_file = config["files"]["pageview_file"]
    timestamp_weights_file = config["files"]["timestamp_weights_file"]

    # simulation data
    init_data_file = config["simulation"]["init_data_file"]
    stream_edits_file = config["simulation"]["stream_edits_file"]
    stream_questions_file = config["simulation"]["stream_questions_file"]

    # wiki api calls config
    start_time = config["wiki_api_config"]["start_time"]
    end_time = config["wiki_api_config"]["end_time"]

    if args.run_query_recentchanges:
        query_recentchanges_api(start_time, end_time, revisions_dir)

    if args.run_query_doc_versions:
        query_doc_versions(titles_file, start_time, end_time, raw_doc_dir)

    if args.run_recent_changes:
        print("Generating from revisions", revisions_dir)
        changes_df = get_recent_changes(revisions_dir, changes_file)

        print("Generated changes file", changes_file)
        titles_df = get_titles(changes_file, titles_file)
        print("Generated titles file", titles_file)
        edits_df = get_edits(edits_file, changes_file, titles_file)
        print("Generated edits file", edits_file)
        # log_files(run, config)

    # query document versions for list of titles
    if args.run_query_doc_versions:
        if not os.path.exists(raw_doc_dir):
            os.mkdir(raw_doc_dir)
        query_doc_versions(titles_file, start_time, end_time, raw_doc_dir)

    # parse documents
    if args.run_parse_docs:
        if not os.path.exists(parsed_doc_dir):
            os.mkdir(parsed_doc_dir)
        if not os.path.exists(parsed_tmp_dir):
            os.mkdir(parsed_tmp_dir)
        parse_docs(raw_doc_dir, parsed_tmp_dir, parsed_doc_dir, workers=32)

    # generate diffs between document versions
    if args.run_generate_diffs:
        # if not os.path.isdir(diff_dir):
        #    os.mkdir(diff_dir)
        generate_diffs(
            edits_file, titles_file, parsed_doc_dir, diff_dir, revisions_file
        )

    if args.run_get_raw_pageviews:
        get_raw_pageviews(raw_pageview_file, titles_file, "2021-09-04T02:45:57Z", "2021-08-05T02:45:57Z")

    # generate pageviews / compute page weights
    if args.run_get_pageviews:
       get_pageviews(raw_pageview_file, pageview_file, edits_file, timestamp_weights_file)
    #    log_pageview(run, config)

    # get question -- requires kevin's question CSVs
    if args.run_get_questions:
        questions_df = get_questions(raw_questions_file, questions_file)
        print("Generated questions file", raw_questions_file, questions_file)
        # log_questions(run, config)

    # generate simulation data
    if args.run_generate_simulation_data:
        generate_simulation_data(
            questions_file,
            edits_file,
            diff_dir,
            init_data_file,
            stream_edits_file,
            stream_questions_file,
        )
        # log_simulation(run, config)

    # run tests to validate simulation data
    if args.run_check_dataset:
        check_dataset(
            titles_file,
            edits_file,
            init_data_file,
            stream_edits_file,
            stream_questions_file,
            diff_dir,
        )

    # generate embeddings for revids from diffs (make passages)
    # if args.run_generate_embeddings:
    #    generate_embeddings(model_file, diff_dir, embedding_dir)
