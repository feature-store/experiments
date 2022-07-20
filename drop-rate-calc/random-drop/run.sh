set -ex

for frac in 0.25 0.5 0.75; do
    KEEP_FRAC_OVERRIDE=${frac} python ./1-overlay-windows.py
    python ./2-compact-array.py
    python ./3-compute-final-score.py
done