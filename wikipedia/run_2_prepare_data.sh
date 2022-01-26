set -xe

plan_dir=/data/wooders/wiki-plans

for replicas in 16 32
do
for model_runtime in 0.25
do
	for event_policy in "lifo"
	do
		for load_shedding_policy in "always_process"
		do
			for key_policy in "round_robin" "weighted_round_robin"
			do
				python wiki_eval_tmp.py --offline-plan-path ${plan_dir}/plan-${key_policy}_${event_policy}-${load_shedding_policy}-${model_runtime}-100_replicas_${replicas}.json --workers 32
			done
		done
	done
done 
done
