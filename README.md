# Feature Store Benchmark 

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






 



