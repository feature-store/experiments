from random import shuffle
import concurrent.futures
from absl import app, flags
import random
import json

import statistics
import pandas as pd
import os
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import configparser
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL

from workloads.util import read_config, use_dataset, log_dataset



FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_keys",
    default=None,
    help="Total number of keys in dataset",
    required=True,
)

flags.DEFINE_integer(
    "num_points",
    default=None,
    help="Total number of keys in dataset",
    required=True,
)

flags.DEFINE_integer(
    "num_workers",
    default=None,
    help="Total number of keys in dataset",
    required=True,
)



def main(argv):
    raw_data_dir = use_dataset("azure", download=False)
    print(raw_data_dir)
    data_dir = f"{raw_data_dir}/azure_{FLAGS.num_keys}"
    key_dir = f"{raw_data_dir}/key_data"

    all_keys = [k.replace(".csv", "") for k in os.listdir(key_dir)]
    print("Total keys", len(all_keys))


    def filter_keys(key):
        try:
            df = pd.read_csv(os.path.join(key_dir, f"{key}.csv"), index_col=0)
        except Exception as e:
            print(e)
            return None
        if len(df.index) >= FLAGS.num_points: 
            return key
        return None


    keys = []
    p = Pool(FLAGS.num_workers)
    for key in p.map(filter_keys, all_keys): 
        if key is not None: keys.append(key)
    p.close()

    print(f"Filtered down to {len(keys)} keys")

    shuffle(keys)

    keys = keys[:FLAGS.num_keys]
    key_index = range(FLAGS.num_keys)

    os.makedirs(data_dir, exist_ok=True)

    json.dump({keys[i]: i for i in key_index}, open(f"{data_dir}/keys_{FLAGS.num_keys}.json", "w"))


    


if __name__ == "__main__":
    app.run(main)
    
