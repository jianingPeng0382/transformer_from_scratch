#!/bin/bash

# 训练脚本
# 使用说明: bash scripts/run.sh

# 设置随机种子
SEED=42

# 创建必要的目录
mkdir -p checkpoints
mkdir -p results

# 激活conda环境（如果使用）
# conda activate transformer

# 安装依赖（如果需要）
# pip install -r requirements.txt

# 下载NLTK数据（用于BLEU计算）
python3 -c "import nltk; nltk.download('punkt')"

# 下载数据集（如果不存在）
if [ ! -d "./dataset" ] || [ -z "$(ls -A ./dataset)" ]; then
    echo "数据集不存在，开始下载..."
    python3 scripts/download_dataset.py
else
    echo "使用本地数据集: ./dataset"
fi

# 训练模型
if python3 src/train.py \
    --tokenizer bert-base-uncased \
    --max_train_samples 20000 \
    --max_val_samples 5000 \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --batch_size 16 \
    --num_workers 4 \
    --d_model 512 \
    --n_layers 6 \
    --n_heads 8 \
    --d_ff 2048 \
    --dropout 0.1 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --clip_grad 1.0 \
    --scheduler cosine \
    --seed ${SEED} \
    --save_dir ./checkpoints \
    --results_dir ./results \
    --save_freq 5; then
    echo "训练成功完成！"
    
    # 评估模型（使用最佳模型）
    if [ -f "./checkpoints/best_model.pt" ]; then
        echo "开始评估模型..."
        python3 src/evaluate.py \
            --checkpoint ./checkpoints/best_model.pt \
            --tokenizer bert-base-uncased \
            --max_val_samples 5000 \
            --max_src_len 512 \
            --max_tgt_len 128 \
            --batch_size 16 \
            --num_workers 4 \
            --eval_samples 100 \
            --results_dir ./results
        echo "评估完成！"
    else
        echo "警告: 未找到最佳模型文件 ./checkpoints/best_model.pt，跳过评估"
    fi
else
    echo "训练失败，跳过评估"
    exit 1
fi

echo "训练和评估完成！"

