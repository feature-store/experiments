set -ex

#data_dir="/home/ubuntu/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/"
data_dir="./yahoo_eval_data"
output_path="./oracle"

tmp_script=`mktemp`
#for data in `ls $data_dir/A4Benchmark-TS*`
for data in `ls $data_dir/*`
do
    key=`basename $data`
    echo \" python evaluation.py --offline-yahoo-csv-path $data \
        --offline-run-oracle \
        --output-path ${output_path}/${key} \" >> $tmp_script
done

cat $tmp_script | xargs -n 1 -P 36 bash -l -c
#cat $tmp_script | parallel --bar bash -l -c
