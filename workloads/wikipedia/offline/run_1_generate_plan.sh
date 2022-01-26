set -xe

for replicas in 1 2 4 8 16 32
do
	for model_runtime in 0.25 #0.001 0.05 0.01 0.1 1.0 5.0 10.0
	do
		for event_policy in "fifo" 
		do
			for load_shedding_policy in "always_process"
			do
				for key_policy in  "round_robin" 
				do
					python simulate.py --model_runtime $model_runtime --send_rate 100 \
						--event_policy  $event_policy --key_policy $key_policy --load_shedding_policy $load_shedding_policy --num_replicas ${replicas}
				done
			done
		done
	done 
done
