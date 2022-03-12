export NUMEXPR_MAX_THREADS=128

for sleep in 0.1 1
do
    for workers in 8 4 2 1
    do 
        for policy in "key-fifo" "random"
        #for policy in "ml"
        do
            TMPDIR=/data/wooders/tmp python workloads/recsys/recsys_server.py \
                --experiment ml-latest-small \
                --scheduler ${policy} \
                --workers ${workers} \
                --update "user" \
                --sleep ${sleep}
        done
    done
done
