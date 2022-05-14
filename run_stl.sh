set -ex

# note: 10*num workers works, 0.01 sleep
slide_size=288
window_size=864
#source_sleep_per_batch=0.01

# make sure update_throughput < event_throughput
#let update_throughput="$workers*2"
#echo $update_throughput
#echo "scale=2 ; $num_keys/ ($source_sleep_per_batch*$slide_size)" | bc

for num_keys in 200 # 10000
do
for source_sleep_per_batch in 0.01
do
for workers in 32 24 16 8 #20 16 24 #10 8 4 1 
do
for algo in ce rr
do
    python workloads/stl/stl_server.py \
     --scheduler=${algo} \
     --window_size=${window_size}\
     --slide_size=${slide_size}\
     --workers=${workers}\
     --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
     --num_keys=${num_keys}\
     --source_sleep_per_batch ${source_sleep_per_batch}
    ray stop --force
    
    #python workloads/stl/stl_server.py \
    # --scheduler=rr \
    # --window_size=${window_size}\
    # --slide_size=${slide_size}\
    # --workers=${workers}\
    # --azure_database /home/ubuntu/cleaned_sqlite_3_days_min_ts.db \
    # --num_keys=${num_keys}\
    # --source_sleep_per_batch ${source_sleep_per_batch}
    #ray stop --force
done 
done 
done 
done
