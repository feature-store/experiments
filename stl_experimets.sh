export NUMEXPR_MAX_THREADS=128

for slide in 672 384 192 96 48 24 12
do
    for workers in 1 8 6 4 2 
    do 
        for policy in "fifo" "lifo" 
        do
            TMPDIR=/data/wooders/tmp python workloads/stl/stl_server.py \
                --experiment stl-yahoo-A4-keys-100-interval-10000-events-200000-queries-200000 \
                --scheduler ${policy} \
                --window_size 672 \
                --slide_size ${slide}\
                --workers ${workers}
        done
    done
done
python workloads/stl/join_queries.py
