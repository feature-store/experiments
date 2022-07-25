set -ex
# note: 10*num workers works, 0.01 sleep
# slide_size=288
# window_size=864
#source_sleep_per_batch=0.01

# make sure update_throughput < event_throughput
#let update_throughput="$workers*2"
#echo $update_throughput
#echo "scale=2 ; $num_keys/ ($source_sleep_per_batch*$slide_size)" | bc

export RAY_ADDRESS=auto

for source_sleep_per_batch in 0.1
do
for workers in 800 # 24 16 8 #20 16 24 #10 8 4 1
do
for algo in rr # ce # rr
do
    python workloads/stl/stl_server_scale.py \
     --scheduler=${algo} \
     --workers=${workers}\
     --source_sleep_per_batch ${source_sleep_per_batch}
    # ray stop --force
done
done
done

