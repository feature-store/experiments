set -ex

data_dir="/data/wooders/stl/yahoo"

for data in `ls $data_dir/A4/*`
do
    key=`basename $data`
    for slide in 6 12 18 24 48 96 168 192 336 672
    do
        python evaluation.py --offline-yahoo-csv-path $data \
            --offline-plan-path ./result/offline_1_slide/plan/slide_${slide}_plan.json \
            --output-path ./result/offline_1_slide/plan_eval/slide_${slide}_key_${key}
    done
done
