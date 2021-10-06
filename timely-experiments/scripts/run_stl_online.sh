set -xe

TIMESTAMP=$(date +%s)
EXP_DIR="./result/online_1_slide"
EXP="experiment_$TIMESTAMP"

mkdir -p $EXP_DIR

# Turn off redis dump
redis-server --save "" --appendonly no &

cargo run --release -- \
    --source=redis \
    --global_window_size=100 \
    --global_slide_size=48 \
    --seasonality=168 &

python ../stl/stl_online_client.py \
  --experiment_dir $EXP_DIR --experiment_id $EXP \
  --redis_snapshot_interval_s 5 --send_rate_per_key 100 \
  --workload yahoo_csv \
  --yahoo_csv_glob_path "/home/peter/workspace/feature-stores-experiments/data/artifacts/yahoo:v0/A4/*.csv" \
  --yahoo_csv_key_extraction_regex "(\d+).csv"

pkill -9 "redis-server"
pkill -9 "timely-experiments"
