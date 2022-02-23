name=$1

# download source dataset to ./artifact
wandb artifact get ${name} 

# run experiment
python stl_server.py --source_dir ./artifact/${name} --target_dir ./results/

# upload/log result data
wandb artifact put --name ${name}-results --type "results" ./results/${name}

