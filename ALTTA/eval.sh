# /home/jjj1/miniconda3/envs/vos/bin/python eval_all_datasets_ttt.py --config_name stcn_mcc \
# --dataset_name davis \
# --split val \
# --dataset_dir /data4/jjj/AFB-URR/DAVIS \
# --output_dir ./log
# --seed 1234



# SEED=1236
# /home/jjj1/miniconda3/envs/vos/bin/python -u eval_all_datasets_ttt.py --seed $SEED \
# --output_dir ./log \
# --dataset_dir /data4/jjj/video_data/ts_setv2 \
# --split val \
# --config_name stcn_mcc \
# --dataset_name SurgicalVideo \
# --res_path ./log1_ep10_s1236_repeat5


#nohup bash eval.sh >>./log1_ep10_s1236.txt 2>&1 &
#b16lr7it100000
#nohup bash eval.sh >>./log1_ep10_s1234_repeat5_base1100.txt 2>&1 &




#!/bin/bash

# 设置起止种子
start_seed=1236
end_seed=1236

echo "开始执行任务，日志将分别保存在单独的文件中..."

for seed in $(seq $start_seed $end_seed)
do
    # 1. 动态定义当前 seed 对应的日志文件名
    # 按照您的要求：new_log1_ep10_s${seed}_repeat5_base1500.txt
    LOG_FILE="./res/new_log1_ep10_s${seed}_repeat5_base1500.txt"
    LOG_FILE="./tmp_mean1236.txt"
    #LOG_FILE="./ablation/beta1/0.01cons_s${seed}.txt"
    LOG_FILE="./ablation/nocons_loss/s${seed}.txt"
    LOG_FILE="./ablation/lr/s${seed}_lr_1e8.txt"
    LOG_FILE="./ablation/curriculumn/only_times_${seed}.txt"
    LOG_FILE="./ablation/abla_lr/s${seed}.txt"
    LOG_FILE="./ablation/seg_res/s${seed}.txt"
    LOG_FILE="./ablation/no_src/s${seed}.txt"
    echo "正在运行 Seed: $seed"
    echo "日志文件: $LOG_FILE"
    
    # 2. 执行 Python 命令
    # 注意： > "$LOG_FILE" 2>&1 被放在了这里，
    # 这样每个 seed 的输出就会进入它专属的 txt 文件
    
    /home/jjj1/miniconda3/envs/vos/bin/python -u eval_all_datasets_ttt.py \
    --seed $seed \
    --output_dir ./log \
    --dataset_dir /data4/jjj/video_data/ts_setv2 \
    --split val \
    --config_name stcn_mcc \
    --dataset_name SurgicalVideo \
    --res_path ./log1_ep10_s${seed}_repeat5 \
    > "$LOG_FILE" 2>&1
    
    # 这里的 python 命令跑完后，才会进入下一个循环
    echo "Seed $seed 完成。"
    
done

echo "所有任务执行完毕。"


# nocons seed1314 86.43 seed1303 86.41 1287 86.41 1284 86.41