set -ex

data_dir="./yahoo_train_data"
results_dir="/data/wooders/stl/results"

tmp_script=`mktemp`
for key_prio in "lifo" "fifo"
do
for data in `ls $data_dir/*`
do
    key=`basename $data`
    for slide in 6 12 18 24 48 96 168 192 336 672
    do
        echo \" python evaluation.py --offline-yahoo-csv-path $data \
            --offline-plan-path ${results_dir}/plan/${key_prio}_slide_${slide}_plan.json \
            --output-path ${results_dir}/single_key/${key_prio}_slide_${slide}_key_${key} \" >> $tmp_script
    done
done
done

cat $tmp_script | xargs -n 1 -P 144 bash -l -c


#set -ex
#
#data_dir="/data/wooders/stl/yahoo"
#
#for data in `ls $data_dir/A4/*`
#do
#    key=`basename $data`
#    for slide in 6 12 18 24 48 96 168 192 336 672
#    do
#        python evaluation.py --offline-yahoo-csv-path $data \
#            --offline-plan-path ./result/offline_1_slide/plan/slide_${slide}_plan.json \
#            --output-path ./result/offline_1_slide/plan_eval/slide_${slide}_key_${key}
#    done
#done
