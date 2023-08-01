# Feature Store Benchmark 

## Setup 

Run 
```
pip install -r requirements.txt
export PYTHONPATH='.'
```
Check the `config.yml` file and make sure the required files and directories exist. 


(Optional) Setup terraform: 
```
terraform apply;
terraform output -json > config.json;
```

## Experiments 

### recsys (ALS) 
1. Train a model with `python workloads/recsys/train_als.py`. Make sure the split/dataset is set to what you want. 
```
python workloads/recsys/als_train \
    --split 0.5 \
    --dataset "ml-1m" \
    --workers 12 \ 
    --resume [True/False] \ # resume from previous checkpoint
    --download_dataset [True/False] \
```
2. Run streaming inference/updates. Make sure you have the right dataset set in the script.

```
python workloads/recsys/stream_als.py \
    --split 0.5 \
    --dataset "ml-1m" \
    --workers 12 \ 
    --download_dataset [True/False] \ # download exisitng model/data
```

3. Evaluate in `nb/als-plots.ipynb`

### anomaly detection (STL) 
1. Run streaming inference/updates.

```
python workloads/stl/stream_simulation.py
```


## Repository structure 

```
experiment_name/ 
    notebooks/ 
    data/ 
    preprocessing/
    simulation/
    ralf/
	client.py
	server.py
    download_data.sh 
```


## Dataset Structure 

Event stream data: `events_<NUM_KEYS>_<TIME_INTERVAL_MS>_<NUM_ROWS>.csv`
* `event_id` (unique id)
* `key_id` 
* `ts` (millisecond timestamp since interval start) 
* value 

Query stream data: `queries_<NUM_KEYS>_<TIME_INTERVAL_MS>_<NUM_ROWS>.csv`
* `query_id` (unique id)
* `key_id` (queried key)
* `ts` 

Optimal feature data: `features_<NUM_KEYS>_<TIME_INTERVAL_MS>_<NUM_ROWS>.csv`
* `key_id` 
* `ts`  (Range from `0-<TIME_INTERVAL_MS>)
* `feature` (optimal pre-computed feature value at `ts`) 

Optimal prediction data `predictions_<NUM_KEYS>_<TIME_INTERVAL_MS>_<NUM_ROWS>.csv`
* `query_id` (corresponds to prediction query)
* `key_id`
* `prediction` (optimal prediction result)

## Experiment Output 
Experiments should output a `actual_features_<...>.csv` and `actual_predictions_<...>.csv` files, which can be compared to pre-generated ideal feature/prediction data to evaluate performance. 






 



