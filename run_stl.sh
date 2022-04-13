set -ex

python workloads/stl/stl_server.py \
 --scheduler=ce \
 --window_size=864 \
 --slide_size=288 \
 --workers=16 \
 --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
 --num_keys=5000 \
 --source_sleep_per_batch 0.001
