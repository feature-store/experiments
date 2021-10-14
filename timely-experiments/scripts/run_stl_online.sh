set -xe

TIMESTAMP=$(date +%s)
EXP_DIR="./result/online_1_slide"
EXP="experiment_$TIMESTAMP"

NUM_PROCESSES=3

mkdir -p $EXP_DIR

# Turn off redis dump
redis-server --save "" --appendonly no &

for ((process_index=0; process_index<$NUM_PROCESSES; process_index++))
do
  cargo run --release -- \
      --source=redis \
      --global_window_size=672 \
      --global_slide_size=48 \
      --per_key_slide_size=./result/min_loss_plan.json \
      --threads=1 \
      --num_processes=$NUM_PROCESSES \
      --process_index=$process_index \
      --seasonality=168 
      --prioritization=lifo &
done

python ../stl/stl_online_client.py \
  --experiment_dir $EXP_DIR --experiment_id $EXP \
  --redis_snapshot_interval_s 5 --send_rate_per_key 100 \
  --workload yahoo_csv \
  --yahoo_csv_glob_path "/home/peter/workspace/feature-stores-experiments/data/artifacts/yahoo:v0/A4/*.csv" \
  --yahoo_csv_key_extraction_regex "(\d+).csv"

pkill -9 "redis-server"
pkill -9 "timely-experiments"
