set -ex

num_keys=20
workers=1
slide_size=288
window_size=864
source_sleep_per_batch=0.01

# make sure update_throughput < event_throughput
let update_throughput="$workers*2"
echo $update_throughput
echo "scale=2 ; $num_keys/ ($source_sleep_per_batch*$slide_size)" | bc

python workloads/stl/stl_server.py \
 --scheduler=ce \
 --window_size=${window_size}\
 --slide_size=${slide_size}\
 --workers=${workers}\
 --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
 --num_keys=${num_keys}\
 --source_sleep_per_batch ${source_sleep_per_batch}

python workloads/stl/stl_server.py \
 --scheduler=rr \
 --window_size=${window_size}\
 --slide_size=${slide_size}\
 --workers=${workers}\
 --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
 --num_keys=20 \
 --epsilon 0.1 \
 --source_sleep_per_batch 0.01


python workloads/stl/stl_server.py \
 --scheduler=ce \
 --window_size=864 \
 --slide_size=288 \
 --workers=16 \
 --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
 --num_keys=20 \
 --epsilon 100 \
 --source_sleep_per_batch 0.01
