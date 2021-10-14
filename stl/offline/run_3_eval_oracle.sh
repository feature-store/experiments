set -ex

# data_dir="/home/ubuntu/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/"
data_dir="/home/ubuntu/yahoo-extended/A4/"

tmp_script=`mktemp`
# for data in `ls $data_dir/A4Benchmark-TS*`
for data in `ls $data_dir/*.csv`
do
    key=`basename $data`
    echo python evaluation.py --offline-yahoo-csv-path $data \
        --offline-run-oracle true \
        --output-path ./result/offline_1_slide/plan_eval/oracle_key_${key} >> $tmp_script
done

cat $tmp_script | parallel --bar bash -l -c
