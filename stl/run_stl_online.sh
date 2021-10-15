set -xe

main() {
    TIMESTAMP=$(date +%s)
    EXP_DIR="./result/online_1_slide"
    EXP="experiment_$TIMESTAMP"
    # Turn off redis dump
    redis-server --save "" --appendonly no &
    
    python stl_online_server.py \
    --experiment_dir $EXP_DIR \
    --experiment_id $EXP \
    --global_slide_size ${OVERRIDE_STATIC_SLIDE_SIZE:-48} \
    --per_key_slide_size_plan_path ${OVERRIDE_PLAN_PATH:-"./offline/result/offline_1_slide/min_loss_plan.json"} \
    ${POLICY_FLAGS} \
    --seasonality 168 \
    --window_size 336 \
    --source_num_replicas 1 \
    --window_num_replicas 4 \
    --map_num_replicas ${OVERRIDE_MAPPER_REPLICAS:-4} \
    --sink_num_replicas 2 \
    &
    
    python stl_online_client.py \
    --experiment_dir $EXP_DIR \
    --experiment_id $EXP \
    --redis_snapshot_interval_s 2 \
    --send_rate_per_key 15 \
    --workload yahoo_csv \
    --yahoo_csv_glob_path '/home/ubuntu/yahoo-extended/A4/*.csv' \
    --yahoo_csv_key_extraction_regex '(\d+).csv'
    
    # Ray stop should also stop that redis-server
    ray stop --force
    pkill -9 "ray"
    # Sleep to make sure ports are cleaned up
    sleep 10
    
    python stl_online_eval.py \
    --experiment_dir ${EXP_DIR}/${EXP} \
    --send_rate_per_key 15 \
    --oracle_csv_glob_path="/home/ubuntu/experiments/stl/offline/result/offline_1_slide/plan_eval/oracle_key_*.csv" \
    --oracle_csv_extraction_regex=".*oracle_key_(\d+).csv"
}

for n_trials in 1 # 2 3
do
    for mapper_replicas in 1 # 2 4 6
    do
        # for static_window in 24 48 96 144 # 12 24 36 48 96 108 120
        # for static_window in 24 48 96 144 196 256 336
        # do
            # # Do once for static window
            # OVERRIDE_STATIC_SLIDE_SIZE=$static_window \
            # OVERRIDE_MAPPER_REPLICAS=$mapper_replicas \
            # main
        # done
        
        # for max_n_fits in 1000 1200 1500 1800 2000 2500
        for max_n_fits in 1000 2000
        do
            # Do once for dynamic window
            POLICY_FLAGS="--use_per_key_slide_size_plan" \
            OVERRIDE_PLAN_PATH="./offline/result/offline_1_slide/min_loss_plan_max_fits_$max_n_fits.json" \
            OVERRIDE_MAPPER_REPLICAS=$mapper_replicas \
            main
        done
    done
done
