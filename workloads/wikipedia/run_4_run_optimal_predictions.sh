set -xe

plan_dir=/data/wooders/wiki-plans
dpr_dir=~/DPR
python wiki_eval.py --offline-plan-path optimal_plan.json
cd $dpr_dir
echo $plan_file
CUDA_VISIBLE_DEVICES=5 bash ${dpr_dir}/evaluate_retrieval_single_doc_stream.sh optimal_plan
