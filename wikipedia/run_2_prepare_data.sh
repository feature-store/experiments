set -xe

plan_dir=/data/wooders/wiki-plans

for key_policy in "round_robin" "weighted_round_robin" #"random" "weighted_random" 
do
	for event_policy in "lifo" "fifo" 
	do
		for load_shedding_policy in "always_process"
		do
			for model_runtime in 0.01 0.05 0.1 1 5 10
			do
				python wiki_eval.py --offline-plan-path ${plan_dir}/plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100.json
			done
		done
	done
done 
p
