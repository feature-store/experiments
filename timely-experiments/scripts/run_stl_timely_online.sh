set -xe

main() {
    TIMESTAMP=$(date +%s)
    EXP_DIR="./result/online_1_slide"
    EXP="experiment_$TIMESTAMP"
    
    mkdir -p $EXP_DIR
    
    # Turn off redis dump
    redis-server --save "" --appendonly no &
    
    cargo build --release
    
    # Don't cargo run here because we immediately send off the client
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
    target/release/timely-experiments \
    --experiment_dir $EXP_DIR \
    --experiment_id $EXP \
    --source=redis \
    --global_window_size=336 \
    --global_slide_size=${OVERRIDE_STATIC_SLIDE_SIZE:-48}  \
    --per_key_slide_size_plan_path=${OVERRIDE_PLAN_PATH:-"./offline/result/offline_1_slide/min_loss_plan.json"} \
    ${POLICY_FLAGS} \
    --threads=${OVERRIDE_MAPPER_REPLICAS:-1} \
    --seasonality=168 &
    
    timely_server_pid=$!
    
    python ../stl/stl_online_client.py \
    --experiment_dir $EXP_DIR \
    --experiment_id $EXP \
    --redis_snapshot_interval_s 2 \
    --send_rate_per_key 10 \
    --workload yahoo_csv \
    --yahoo_csv_glob_path '/home/ubuntu/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS*.csv' \
    --yahoo_csv_key_extraction_regex 'A4Benchmark-TS(\d+).csv'
    
    kill -9 $timely_server_pid
    pkill -9 "redis-server"
    
    python ../stl/stl_online_eval.py \
    --experiment_dir ${EXP_DIR}/${EXP} \
    --send_rate_per_key 10 \
    --is_timely_result
}


for n_trials in 1 2 3
do
    for mapper_replicas in 1 2 4 6
    do
        for static_window in 24 48 96 144 # 12 24 36 48 96 108 120
        do
            # Do once for static window
            OVERRIDE_STATIC_SLIDE_SIZE=$static_window \
            OVERRIDE_MAPPER_REPLICAS=$mapper_replicas \
            main
        done
        
        for max_n_fits in 1000 1200 1500
        do
            # Do once for dynamic window
            POLICY_FLAGS="--use_per_key_slide_size_plan" \
            OVERRIDE_PLAN_PATH="../stl/offline/result/offline_1_slide/min_loss_plan_max_fits_$max_n_fits.json" \
            OVERRIDE_MAPPER_REPLICAS=$mapper_replicas \
            main
        done
    done
done