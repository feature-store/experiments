set -ex

python simulation.py --model_runtime_s 0.02 --total_runtime_s 150 \
    --per_key_records_per_second 100 \
    --num_mapper_replicas 2 --num_keys 100 \
    --window_size 672 --slide_size 0 \
    --per_key_slide_size_plan result/offline_1_slide/min_loss_plan.json \
    --output_path result/offline_1_slide/lp_eval/varying_slide_size_trace.json