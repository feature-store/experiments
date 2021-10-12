set -xe

for key_policy in  "weighted_random" "weighted_round_robin"
do
	for event_policy in "lifo" 
	do
		for load_shedding_policy in "always_process"
		do
			for model_runtime in 0.01 0.05 0.1 1 5 10
			do
				python simulate.py --model_runtime $model_runtime --send_rate 100 \
					--event_policy  $event_policy --key_policy $key_policy --load_shedding_policy $load_shedding_policy
			done
		done
	done
done 
