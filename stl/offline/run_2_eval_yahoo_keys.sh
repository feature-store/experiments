set -ex

data_dir="/home/ubuntu/yahoo-extended/A4/"

tmp_script=`mktemp`
for data in `ls $data_dir/*.csv`
do
    key=`basename $data`
    # for slide in 2 6 12 24 36 48 96 108 120 144
    for slide in 2 6 12 24 36 48 96 108 120 144 196 220 256 284 308 336
    do
        echo python evaluation.py --offline-yahoo-csv-path $data \
        --offline-plan-path ./result/offline_1_slide/plan/slide_${slide}_plan.json \
        --output-path ./result/offline_1_slide/plan_eval/slide_${slide}_key_${key} >> $tmp_script
    done
done

cat $tmp_script | parallel --bar bash -l -c
