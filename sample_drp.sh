
#!/bin/bash
# # echo "10:00"
# # sleep 14400

# # echo "3:00"
# # sleep 10800

# # echo "2:00"
# # sleep 1800
# # echo "1:30"
# # sleep 1800
# # echo "1:00"
# # sleep 1800
# # echo "0:30"
# # sleep 900

# echo "START"
# CUDA_VISIBLE_DEVICES=2 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.0 &
# CUDA_VISIBLE_DEVICES=2 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.1 &
# CUDA_VISIBLE_DEVICES=2 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.2 &
# CUDA_VISIBLE_DEVICES=2 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.3 &
# CUDA_VISIBLE_DEVICES=2 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.4 &
# CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.5 &
# CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.6 &
# CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.7 &
# CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.8 &
# CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.9 &


# CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.0 &
# CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.2 &
# CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.5 &
# CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 2.0 &
# CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 2.5 &
# CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 5.0 &
# CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 7.0 &
# CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 10 &
# CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 12 &
# CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 15 &

# # CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.0 &
# # CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.1 &
# # CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.2 &
# # CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.3 &
# # CUDA_VISIBLE_DEVICES=0 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.4 &
# # CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.6 &
# # CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.8 &
# # CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 0.9 &
# # CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.0 &
# # CUDA_VISIBLE_DEVICES=1 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.5 &


# # CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 1.7 &
# # CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 3.0 &
# # CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 2.0 &
# # CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 2.5 &
# # CUDA_VISIBLE_DEVICES=3 python main.py --type frag_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_clfg_zinc2050k.yaml --condition 3.0 &



# echo "END"


degree=5
for i in `seq 0 19`
do
    # sleep 1 & # 提交到后台的任务
    # echo $i && sleep 1 &
    CUDA_VISIBLE_DEVICES=0 python main.py --type drp_condition_sample --config /home/lk/project/mol_generate/GDSS/config/sample_gdscv2.yaml --condition  "$((i / 10)).$((i % 10))" &
    [ $(( (i+1) % degree )) -eq 0 ] && wait
done


