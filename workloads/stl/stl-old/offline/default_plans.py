import json

plan_dir = "/data/wooders/stl/results"
slides = [1, 6, 12, 18, 24, 48, 96, 168, 192, 336, 672]

for slide in slides: 
    weights = {i: slide for i in range(1, 101, 1)}
    open(f"{plan_dir}/plan_baseline_{slide}.json", "w").write(json.dumps(weights))
