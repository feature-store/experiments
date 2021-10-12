set -xe

plan_dir=/data/wooders/wiki-plans
dpr_dir=/home/eecs/wooders/DPR
wiki_dir=/home/eecs/wooders/experiments/wikipedia


#for key_policy in "weighted_random" "weighted_round_robin" 
#for key_policy in "random" "weighted_random"
for key_policy in "round_robin" "weighted_round_robin" 
do
	for event_policy in "lifo"
	do
		for load_shedding_policy in "always_process"
		do
			for model_runtime in 0.01 0.05 0.1 1 5 
			do
				cd $wiki_dir
				plan_file=plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100
				echo $plan_file
				python wiki_eval.py --offline-plan-path ${plan_dir}/${plan_file}.json
				cd $dpr_dir
				CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash ${dpr_dir}/evaluate_retrieval_single_doc_stream.sh $plan_file & 
				pid=$!
			done
			#wait $pid
		done
	done
done 
p
