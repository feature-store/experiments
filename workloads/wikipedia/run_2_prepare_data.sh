set -xe

plan_dir=/data/jeffcheng1234/ralf-vldb/results/wikipedia/plans

for replicas in 1
do
#for model_runtime in 1000000.0 #1e-06 0.001 0.05 0.01 0.1 1.0 5.0 10.0
#for model_runtime in 0.01 0.25 1.0 10.0 #1000000.0 #1e-06   0.001 0.05 0.01 0.1 1.0 5.0 10.0
for model_runtime in 0.001 0.05 0.01 0.1 1.0 5.0 10.0
do
	for event_policy in "lifo"
	do
		for load_shedding_policy in "always_process"
		do
			for key_policy in "round_robin" #"weighted_round_robin"
			do
				#python wiki_eval_tmp.py --offline-plan-path ${plan_dir}/plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100_replicas_${replicas}.json --workers 32
				python ~/experiments/workloads/wikipedia/prepare_dpr_data.py --offline-plan-path ${plan_dir}/plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100_replicas_${replicas}.json --workers 32
			done
		done
	done
done 
done
