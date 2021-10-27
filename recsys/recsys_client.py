import sys
from tqdm import tqdm
import argparse
import os
import json
import time

from threading import Timer

import psutil

from ralf.client import RalfClient

client = RalfClient()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Specify experiment config")

    # Experiment related
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/sarahwooders/repos/flink-feature-flow/datasets",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="/Users/sarahwooders/repos/flink-feature-flow/RayServer/experiments",
    )
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    
    #user_id = "1"
    #res = client.point_query(key=user_id, table_name="user_vectors")
    #print(res)
    res = client.bulk_query(table_name="user_vectors")
    print([r for r in res])