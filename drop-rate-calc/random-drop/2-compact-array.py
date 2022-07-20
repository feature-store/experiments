import os
import time
import numpy as np
from tqdm import tqdm
from glob import glob
import yaml

config = yaml.safe_load(open("config.yaml"))


input_dir = f"{config['materialized_prediction_output_path']}/*.npy"
output_file = config["compact_prediction_output_path"]


arrays = []
for path in tqdm(glob(input_dir)):
    arrays.append((int(os.path.split(path)[1].replace(".npy", "")), np.load(path)))

print("begin sorting", time.time())
arrays.sort(key=lambda tup: tup[0])
print("end sorting", time.time())

compact = np.stack([a[1] for a in arrays]).astype("float32")
print(compact.shape)
np.save(output_file, compact)
