# ALS Workload 

## Preparing Data 
Run `python workloads/recsys/generate_data.py` (make sure you're using the right MovieLens directory)

## Training Initial Embeddings 
To run on GPU, run: 
```
CUDA_VISIBLE_DEVICES=2,3 python workloads/recsys/als.py
``` 
(TODO: add flag for CPU)

## Running `ralf`
Run:
```
python workloads/recsys/recsys_server.py \
    --experiment ml-100k-features \ # dataset name 
    --update user \ # which feature update function to use (user/als/sgd)
    --scheduler fifo \ # scheduler parameters
    --sleep 0.1 \ # sleep time between timesteps for DataSource
    --workers 1 \ # number of ray replicas
```
The script will automatically read the embeddings and data from prior steps. 
