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
    config.read("config.yml")
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
    cred = read_credentials()
    aws_access_key_id = cred["aws_access_key_id"]
    aws_secret_access_key = cred["aws_secret_access_key"] 

    # download form s3
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket("feature-store-datasets")
    objs = list(bucket.objects.folder(Prefix = f"{source_dir}/{name}")) 
    if len(objs) == 0: 
        print("Bucket is empty or does not exist", bucket.objects.folder(Prefix = source_dir))
        return None

    # create local directory
    if not os.path.isdir(os.path.join(target_dir, name)):
        os.mkdir(os.path.join(target_dir, name))

    # download objects
    for obj in objs: 
        print(obj.key, "target", os.path.join(target_dir, name, obj.key.replace(source_dir, "")))
        bucket.download_file(obj.key, os.path.join(target_dir, name, obj.key.replace(source_dir, "")))

    return os.path.join(target_dir, name)

def upload_dir(name, source_dir, target_dir): 
    cred = read_credentials()
    aws_access_key_id = cred["aws_access_key_id"]
    aws_secret_access_key = cred["aws_secret_access_key"] 

    print(aws_access_key_id, aws_secret_access_key)
    # download form s3
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    #bucket = s3.Bucket("feature-store-datasets")

    print(f"{target_dir}/{name}/f")
    for root,dirs,files in os.walk(os.path.join(source_dir, name)):
        for f in files: 
            print("uploading", root, f)
            s3.upload_file(os.path.join(root, f), "feature-store-datasets", f"{target_dir}/{name}/f")

    return f"{target_dir}/{name}"

def use_dataset(name, redownload = False):
    config = read_config()
    
    if not os.path.isdir(path) or redownload: 
        # download form s3
        print("Downloading from aws:", config["aws_dir"])
        return download_dir(name, config["aws_dir"] + "/datasets", config["dataset_dir"])

    return os.path.join(config["dataset_dir"], name)

def use_results(name, redownload = False):
    config = read_config()
    
    if not os.path.isdir(path) or redownload: 
        # download form s3
        print("Downloading from aws:", config["aws_dir"])
        return download_dir(name, config["aws_dir"] + "/results", config["results_dir"])

    return os.path.join(config["results_dir"], name)

def log_dataset(name):
    config = read_config()
    return upload_dir(name, config["dataset_dir"], config["aws_dir"] + "/datasets")

def log_results(name):
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
        self.file = None

    @property
    def _file(self):
        if self.file is None:
            self.file = open(self.filename, "a")
        return self.file

    def on_event(self, record: Record): 
        print({col: [getattr(record.entry, col)] for col in self.cols})
        df = pd.DataFrame({col: [getattr(record.entry, col)] for col in self.cols})
        self._file.write(df.T.to_csv(index=None, header=None))
        open(self.filename, "a").write(df.T.to_csv(index=None, header=None))
       
