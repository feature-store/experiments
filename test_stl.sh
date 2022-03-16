export NUMEXPR_MAX_THREADS=128

for slide in 1
do
    for workers in 1
    do 
        for policy in "rr" "ce" #"fifo"
        do
            TMPDIR=/data/wooders/tmp python workloads/stl/stl_server.py \
                --experiment "yahoo/A1" \
                --scheduler ${policy} \
                --window_size 48 \
                --slide_size ${slide}\
                --workers ${workers}
        done
    done
done
#python workloads/stl/join_queries.py
