set -xe

plan_dir=/data/wooders/wiki-plans
dpr_dir=~/DPR
cd $dpr_dir
plan_file="optimal_plan"
echo $plan_file
CUDA_VISIBLE_DEVICES=5 bash ${dpr_dir}/evaluate_retrieval_single_doc_stream.sh $plan_file
