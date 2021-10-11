set -ex

# TODO(simon): use a workflow engine for step tracking
# e.g. https://dagster.io/

for max_n_fits in 1000 1200 1500
do
    python config_gen.py \
    --csv_dir "./result/offline_1_slide/plan_eval" \
    --output_path "./result/offline_1_slide/min_loss_plan_max_fits_$max_n_fits.json" \
    --max_n_fits $max_n_fits
done