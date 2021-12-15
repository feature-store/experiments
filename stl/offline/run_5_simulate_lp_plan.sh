set -ex
#PARAM_DIR="offline_1_slide"
PARAM_DIR="/data/wooders/stl/results"
PLAN_DIR="/data/wooders/stl/results"
#OUTPUT_CSV_PATH="offline_1_slide/lp_plan_eval"
#TRAIN_PATH="./yahoo_train_data/data/wooders/stl/results"
#EVAL_PATH="./yahoo_eval_data"
EVAL_PATH="/home/eecs/wooders/experiments/stl/notebooks/artifacts/yahoo_eval_data:v0"
TRAIN_PATH="/home/eecs/wooders/experiments/stl/notebooks/artifacts/yahoo_train_data:v0"


for replicas in 8
do
for plan in "max_fits_1100" "max_fits_2100" "max_fits_4200" "max_fits_8400"
do
	mkdir -p ${PLAN_DIR}/replica_${replicas}

	# re-run simulation with lp-generated weights
	python simulation.py --model_runtime_s 1.5 --total_runtime_s 2000 \
	    --per_key_records_per_second 1 \
	    --num_mapper_replicas ${replicas} \
	    --window_size 672 --slide_size 0 \
	    --per_key_slide_size_plan ${PARAM_DIR}/${plan}.json \
	    --output_path ${PLAN_DIR}/replica_${replicas}/plan_${plan}.json \
	    --source_data_path ${TRAIN_PATH}
	
	mkdir -p ${PLAN_DIR}/replica_${replicas}/${plan}

	# run evaluation with simulation results
	python evaluation.py --offline-yahoo-csv-path $EVAL_PATH \
	 	--offline-plan-path ${PLAN_DIR}/replica_${replicas}/plan_${plan}.json \
		--output-path ${PLAN_DIR}/replica_${replicas}/${plan} \
		--param-path ${PARAM_DIR}/${plan}.json \
		--run-policy 
	
	# get final results
	#python evaluate_loss.py --offline-yahoo-csv-path $SOURCE_PATH --predicted-csv-path $OUTPUT_CSV_PATH --output-path 
done	
done
