set -xe

for slide in 2 6 12 24 36 48 96 108 120 144 196 220 256 284 308 336
do
    python simulation.py --model_runtime_s 0 --total_runtime_s 2000 --per_key_records_per_second 1 \
    --window_size 336 --slide_size ${slide} --output_path result/offline_1_slide/plan/slide_${slide}_plan.json
done
