set -ex

# TODO(simon): use a workflow engine for step tracking
# e.g. https://dagster.io/

#python config_gen.py \
#    --csv_dir "./result/offline_1_slide/plan_eval" \
#    --output_path "./result/offline_1_slide/min_loss_plan.json"

MAX_FITS=500
python config_gen.py \
    --csv_dir "/data/wooders/stl/results/single_key" \
    --output_path "/data/wooders/stl/results/test_${MAX_FITS}.json" \
    --max_n_fits ${MAX_FITS}
    --csv_dir "./result/offline_1_slide/plan_eval" \
    --output_path "./result/offline_1_slide/min_loss_plan.json"
