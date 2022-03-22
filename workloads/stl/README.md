# STL Experiment Pipeline 
 
Run scripts from repo parent directory. Config for expeirments is at `../../config.yml` and credentials for AWS should be stored at `../../ralf-vldb.json`

# Yahoo Data

## Generate Dataset 
```
python workloads/stl/generate_data.py \
    --num_keys 100 \
    --time_interval_ms 2000 \
    --num_events 200000 \
    --num_queries 1000
```

## Run Experiment
```
python workloads/stl/stl_server.py \
    --experiment stl-yahoo-A4-keys-100-interval-10000-events-200000-queries-200000 \
    --scheduler lifo \
    --window_size 672 \
    --slide_size 96 \
    --workers 4 \
```

# Azure Data 
The Azure data is the V2 dataset downloaded from `https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetLinksV2.txt`. 

The dataset (according to the paper) has the following properties: 
* 2.6 million keys 
* 1.9B utilization readings (NOT uniformly distributed over keys) 
* Readings every 5 minutes (`timestamp` column is in seconds) 
* Daily seasonality (24 hours = 288 points) 
* Recommended 3 day window (72 hours = 864 points)

## Downloading Data
Make sure you have `../../config.yml` set to where you want data saved. You can download the data by running `use_dataset("azure", download=True)` or by copying `s3://feature-store-datasets/vldb/datasets/azure/` directly. 

The dataset has the following structure: 
```
azure/
    raw_data/ # decompressed CSV files index 1-197
    tmp/ # CSV files split by key (map)
    key_data/ # CSV for each key (reduce)
```
You can run `python read_keys_azure.py` to process the raw data. 

## Preprocessing Data 


## Run Experiment
```
python workloads/stl/stl_server.py \
    --experiment azure/azure_small \
    --scheduler fifo \
    --window_size 864 \
    --slide_size 1 \
    --workers 4 \
```




