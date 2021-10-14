set -xe
PARAM_DIR="/data/wooders/stl/results"
PLAN_DIR="/data/wooders/stl/results"
TRAIN_PATH="./yahoo_train_data"
EVAL_PATH="./yahoo_eval_data"

for key_policy in "fifo"
do
for replicas in 1 2 4 8
do 
	for slide in 672 1 6 12 18 24 48 96 168 192 336 
	do
		plan="plan_baseline_${slide}_${key_policy}"
		param="plan_baseline_${slide}"
		mkdir -p ${PLAN_DIR}/replica_${replicas}
		python simulation.py \
			--model_runtime_s 1.5 \
			--total_runtime_s 2000 \
			--per_key_records_per_second 1 \
			--window_size 672 \
			--slide_size ${slide} \
			--per_key_slide_size_plan ${PARAM_DIR}/${param}.json \
			--output_path ${PLAN_DIR}/replica_${replicas}/${plan}.json \
			--source_data_path $TRAIN_PATH \
			--num_mapper_replicas ${replicas} \
			--key_prio_policy ${key_policy}

		mkdir -p ${PLAN_DIR}/replica_${replicas}/${plan}
		python evaluation.py --offline-yahoo-csv-path $EVAL_PATH \
			--offline-plan-path ${PLAN_DIR}/replica_${replicas}/${plan}.json \
			--output-path ${PLAN_DIR}/replica_${replicas}/${plan} \
			--param-path ${PARAM_DIR}/${param}.json \
			--run-policy 

	done
done
done
