set -ex

data_dir="./yahoo_train_data"
tmp_script=`mktemp`

for key_prio in "lifo" "fifo"
do
for data in `ls $data_dir/*`
do
    key=`basename $data`
    for slide in 6 12 18 24 48 96 168 192 336 672
    do
 	echo \" python simulation.py --model_runtime_s 1.5 --total_runtime_s 2000 --per_key_records_per_second 1 --key_prio_policy ${key_prio} --window_size 672 --slide_size ${slide} --output_path offline_1_slide/plan/${key_prio}_slide_${slide}_plan.json --num_mapper_replicas 1\" >> $tmp_script
    done
done
done

cat $tmp_script | xargs -n 1 -P 36 bash -l -c

#set -xe
#
#for replicas in 
#for slide in 1 6 12 18 24 48 96 168 192 336 672
#do
#    python simulation.py --model_runtime_s 0 --total_runtime_s 2000 --per_key_records_per_second 1 \
#        --window_size 672 --slide_size ${slide} --output_path result/offline_1_slide/plan/slide_${slide}_plan.json
#done
