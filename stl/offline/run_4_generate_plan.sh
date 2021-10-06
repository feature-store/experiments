set -ex

# TODO(simon): use a workflow engine for step tracking
# e.g. https://dagster.io/

python config_gen.py \
    --csv_dir "./result/offline_1_slide/plan_eval" \
    --output_path "./result/offline_1_slide/min_loss_plan.json"