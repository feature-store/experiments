set -xe

dpr_dir=~/DPR

cd $dpr_dir

for replicas in 1
do
for event_policy in "lifo"
do
    for model_runtime in 0.01 0.25 1.0 #10.0 #1000000.0 #1e-06   0.001 0.05 0.01 0.1 1.0 5.0 10.0
	#for model_runtime in 0.01 0.05 0.1 1.0 10.0 0.25 0.005
	do
		for load_shedding_policy in "always_process"
		do
			for key_policy in  "round_robin" 
			do
				#plan_file=plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100
				plan_file=plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100_replicas_${replicas}
				echo $plan_file
				CUDA_VISIBLE_DEVICES=1 bash ${dpr_dir}/evaluate_retrieval_single_doc_stream.sh $plan_file

				#pid=$!
			done
			#wait $pid
		done
	done
done 
done
