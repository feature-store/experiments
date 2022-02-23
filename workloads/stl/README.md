# STL Experiment Pipeline 
 
Run scripts from repo parent directory. Config for expeirments is at `../../config.yml` and credentials for AWS should be stored at `../../ralf-vldb.json`

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


