set -ex
PARAM_PATH=result/offline_1_slide/min_loss_plan.json
PLAN_PATH=result/offline_1_slide/lp_eval/varying_slide_size_trace.json
SOURCE_PATH=/data/wooders/stl/yahoo/A4
OUTPUT_CSV_PATH=result/offline_1_slide/

# re-run simulation with lp-generated weights
python simulation.py --model_runtime_s 0.02 --total_runtime_s 150 \
    --per_key_records_per_second 100 \
    --num_mapper_replicas 2 \
    --window_size 672 --slide_size 0 \
    --per_key_slide_size_plan $PARAM_PATH \
    --output_path $PLAN_PATH \
    --source_data_path $SOURCE_PATH

# run evaluation with simulation results
python evaluation.py --offline-yahoo-csv-path $SOURCE_PATH \
 	--offline-plan-path $PLAN_PATH  \
	--output-path $OUTPUT_CSV_PATH \ 
	--param-path $PARAM_PATH \
	--run-policy 

# get final results
python evaluate_loss.py --offline-yahoo-csv-path $SOURCE_PATH --predicted-csv-path $OUTPUT_CSV_PATH --output-path 


