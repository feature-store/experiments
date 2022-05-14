import os
import json
import boto3
import configparser
import pandas as pd
from typing import List
from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
# TODO: Common source operator to ingest events.csv


def read_config(): 
    config = configparser.ConfigParser()
    # TODO: change
    config.read("/home/ubuntu/workspace/experiments/config.yml")
    return { 
        "results_dir": config["default"]["results_dir"], 
        "dataset_dir": config["default"]["dataset_dir"], 
        "aws_dir": config["default"]["aws_dir"],
        "credentials": config["default"]["credentials"]
    }

def read_credentials(): 
    filename = read_config()["credentials"]
    return json.load(open(filename))
 
def download_dir(name, source_dir, target_dir):
    """
    Download directory from s3 

    :name: directory name (folder/expeirment name)
    :source_dir: dataset/results local directory 
    :target_dir: dataset/results folder in s3
    """

    cred = read_credentials()
    aws_access_key_id = cred["aws_access_key_id"]
    aws_secret_access_key = cred["aws_secret_access_key"] 

    # download form s3
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    objs  = s3.list_objects(Bucket="feature-store-datasets", Prefix=f"{source_dir}/{name}")
    if len(objs) == 0: 
        print("Bucket is empty or does not exist", s3.list_objects(Bucket="feature-store-datasets", Prefix=f"{source_dir}"))
        return None

    # create local directory
    if not os.path.isdir(os.path.join(target_dir, name)):
        os.mkdir(os.path.join(target_dir, name))

    # download objects
    if 'Contents' not in objs: 
        print("Cannot download")
        return os.path.join(target_dir, name)

    for obj in objs['Contents']: 
        key = obj['Key']
        target = target_dir + key.replace(source_dir, "")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        s3.download_file("feature-store-datasets", key, target)

    return os.path.join(target_dir, name)

def upload_dir(name, source_dir, target_dir): 
    """
    Upload directory to s3

    :name: directory name (folder/expeirment name)
    :source_dir: dataset/results local directory 
    :target_dir: dataset/results folder in s3
    """
    cred = read_credentials()
    aws_access_key_id = cred["aws_access_key_id"]
    aws_secret_access_key = cred["aws_secret_access_key"] 

    print("uploading", source_dir, name)

    # download form s3
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    for root,dirs,files in os.walk(os.path.join(source_dir, name)):
        for f in files: 
            key = os.path.join(root, f).replace(source_dir, "")
            print(target_dir, name, key)
            target = target_dir + key
            print("uploading", root, target_dir, target)
            assert os.path.exists(os.path.join(root, f))
            s3.upload_file(os.path.join(root, f), "feature-store-datasets", target)

    return f"{target_dir}/{name}"

def use_dataset(name, download = False):
    config = read_config()
   
    path = os.path.join(config["dataset_dir"], name)
    print(path)
    if download:
        # download form s3
        print("Downloading from aws:", config["aws_dir"])
        return download_dir(name, config["aws_dir"] + "/datasets", config["dataset_dir"])
    elif not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    return os.path.join(config["dataset_dir"], name)

def use_results(name, download = False):
    config = read_config()
    
    path = os.path.join(config["results_dir"], name)
    if download:
        # download form s3
        print("Downloading from aws:", config["aws_dir"])
        return download_dir(name, config["aws_dir"] + "/results", config["results_dir"])
    elif os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    return os.path.join(config["results_dir"], name)

def log_dataset(name):
    """
    Upload dataset folder to s3 bucket 

    :name: folder name (corresponds to experiment name) - should already exists in dataset_dir
    """
    config = read_config()
    return upload_dir(name, config["dataset_dir"], config["aws_dir"] + "/datasets")

def log_results(name):
    """
    Upload results folder to s3 bucket 

    :name: folder name (corresponds to experiment name) - should already exists in results_dir 
    """

    config = read_config()
    return upload_dir(name, config["results_dir"], config["aws_dir"] + "/results")


class WriteFeatures(BaseTransform): 
    """
    ralf transform to log all the feature versions (updates) being updated to a CSV file. 
    The written CSV file can be joined with the queries CSV to determine what the feature "version" would have
    been when each query is made. 

    :filename: file to write features to 
    :columns: list of column names for feature CSV
    """

    def __init__(self, filename: str, columns: List[str]):
        df = pd.DataFrame({col: [] for col in columns})
        self.cols = columns
        self.filename = filename 
        df.to_csv(self.filename, index=None)
        #self.file = None

    #@property
    #def _file(self):
        #if self.file is None:
            #self.file = open(self.filename, "a")
        #return self.file

    def on_event(self, record: Record): 
        df = pd.DataFrame({col: [getattr(record.entry, col)] for col in self.cols})
        df.to_csv(self.filename, mode="a", index=False, header=False)
      


def join_queries_features(queries_df, features_df, time_field="timestamp", key_field="key_id"): 

    # TODO: shard/parallleize by key 

    from tqdm import tqdm 

    ## assing timestamp 
    #features_df["timestamp"] = features_df["processing_time"]
    #start_time = features_df["timestamp"].min()
    #queries_df = assigned_query_timestamps(queries_df, start_time)

    queries_df = queries_df.sort_values([key_field, time_field])
    features_df = features_df.sort_values([key_field, time_field])

    fi = 0

    rows = []
    for qi in tqdm(range(len(queries_df.index))):

        key = queries_df.iloc[qi][key_field]
        ts = queries_df.iloc[qi][time_field]
        #print(fi, qi, features_df.iloc[fi].key_id, key, features_df.iloc[fi].timestamp, ts)

        while fi < len(features_df.index) and features_df.iloc[fi][key_field] != key:
            fi += 1

        while fi + 1 < len(features_df.index) and features_df.iloc[fi + 1][time_field] <= ts and features_df.iloc[fi + 1][key_field] == key: 
            fi += 1

        if fi >= len(features_df.index): break

        if features_df.iloc[fi][time_field] > ts or features_df.iloc[fi][key_field] != key: 

            row = {col: None for col in features_df.iloc[fi].to_dict().keys()}
            row["query_id"] = int(queries_df.iloc[qi].query_id)
            row[f"query_{key_field}"] = queries_df.iloc[qi][key_field]
            rows.append(row)


            continue

        assert features_df.iloc[fi][time_field] <= ts and features_df.iloc[fi][key_field] == key, f"Mismatch {fi}/{qi}: {features_df.iloc[fi].timestamp}/{ts}, {features_df.iloc[fi][key_field]}/{key}"

        row = features_df.iloc[fi].to_dict()
        row["query_id"] = int(queries_df.iloc[qi].query_id)
        row[f"query_{key_field}"] = queries_df.iloc[qi][key_field]
        rows.append(row)

    return pd.DataFrame(rows)



def assigned_query_timestamps(queries_df, start_time, interval_time=1): 
    """

    :start_time: first ingest timestamp from events 
    :interval_time: time between queries 
    """
    
    queries_df["timestamp"] = queries_df["timestamp_ms"].apply(lambda x: x*interval_time + start_time)
    print(queries_df["timestamp_ms"])
    print("timestamp", queries_df["timestamp"])
    return queries_df

def join_queries_features_key(input_queries_df, input_features_df, key): 

    queries_df = input_queries_df[input_queries_df["key_id"] == key].sort_values(["timestamp_ms"])
    features_df = input_features_df[input_features_df["key_id"] == key].sort_values(["timestamp_ms"])

    fi = 0

    rows = []
    for qi in tqdm(range(len(queries_df.index))):

        key = queries_df.iloc[qi].key_id
        ts = queries_df.iloc[qi].timestamp_ms
        #print(fi, qi, features_df.iloc[fi].key_id, key, features_df.iloc[fi].timestamp_ms, ts)

        while fi < len(features_df.index) and features_df.iloc[fi].key_id != key:
            fi += 1

        while fi + 1 < len(features_df.index) and features_df.iloc[fi + 1].timestamp_ms <= ts and features_df.iloc[fi + 1].key_id == key: 
            fi += 1

        if fi >= len(features_df.index): break

        if features_df.iloc[fi].timestamp_ms > ts or features_df.iloc[fi].key_id != key: 
            continue

        assert features_df.iloc[fi].timestamp_ms <= ts and features_df.iloc[fi].key_id == key, f"Mismatch {fi}/{qi}: {features_df.iloc[fi].timestamp_ms}/{ts}, {features_df.iloc[fi].key_id}/{key}"

        row = features_df.iloc[fi].to_dict()
        row["query_id"] = int(queries_df.iloc[qi].query_id)
        row["query_key_id"] = queries_df.iloc[qi].key_id
        rows.append(row)

    return pd.DataFrame(rows)





