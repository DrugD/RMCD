
#!/bin/bash

degree=5
for i in `seq 0 29`
do
    # echo $i
    # sleep 1 & 
    CUDA_VISIBLE_DEVICES=3 python main.py --type dta_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_davis.yaml --condition  "$((i / 10)).$((i % 10))" & 
    [ $(( (i+1) % degree )) -eq 0 ] && wait
done


