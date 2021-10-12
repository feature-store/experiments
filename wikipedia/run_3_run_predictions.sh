set -xe

plan_dir=/data/wooders/wiki-plans
dpr_dir=~/DPR

cd $dpr_dir

for key_policy in "round_robin" "weighted_round_robin" 
#for key_policy in "random" "weighted_random"
do
	for event_policy in "lifo"
	do
		for load_shedding_policy in "always_process"
		do
			for model_runtime in 0.01 0.05 0.1 1 5 
			do
				plan_file=plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100
				echo $plan_file
				CUDA_VISIBLE_DEVICES=1,2,5 bash ${dpr_dir}/evaluate_retrieval_single_doc_stream.sh $plan_file & 
				pid=$!
			done
			#wait $pid
		done
	done
done 
p
