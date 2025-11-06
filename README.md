# Transformer实现 - CNN/DailyMail摘要任务

本项目实现了完整的Transformer模型（Encoder-Decoder架构），并在CNN/DailyMail数据集上进行文本摘要任务的训练和评估。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── transformer.py     # Transformer模型实现
│   ├── dataset.py         # 数据集加载和预处理
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── scripts/               # 脚本目录
│   └── run.sh            # 训练和评估运行脚本
├── results/              # 结果目录（训练曲线、评估结果等）
├── checkpoints/          # 模型检查点目录
├── requirements.txt      # Python依赖
└── README.md            # 本文件
```

## 环境要求

### 硬件要求
- GPU: 建议使用NVIDIA GPU（至少8GB显存）
- 内存: 至少16GB RAM
- 存储: 至少10GB可用空间（用于数据集和模型）

### 软件要求
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0（如果使用GPU）

## 安装步骤

1. 克隆或下载项目代码

2. 创建conda环境（推荐）：
```bash
conda create -n transformer python=3.10
conda activate transformer
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载NLTK数据（用于评估指标）：
```bash
python -c "import nltk; nltk.download('punkt')"
```

## 数据集

本项目使用CNN/DailyMail数据集进行文本摘要任务。数据集存储在本地`./dataset`文件夹中。

- **数据集链接**: [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)
- **任务类型**: 文本摘要（Sequence-to-Sequence）
- **数据规模**: 
  - 训练集: 20,000样本（可配置）
  - 验证集: 5,000样本（可配置）

### 下载数据集

首次运行会自动下载数据集，也可以手动下载：

```bash
python scripts/download_dataset.py
```

数据集将保存到 `./dataset/` 目录中。如果该目录已存在，程序会自动使用本地数据集。

## 模型架构

### 核心组件

1. **Multi-Head Self-Attention**: 多头自注意力机制
2. **Position-wise Feed-Forward Network**: 位置前馈网络
3. **Residual Connections + Layer Normalization**: 残差连接和层归一化
4. **Positional Encoding**: 正弦/余弦位置编码
5. **Encoder-Decoder Architecture**: 完整的编码器-解码器架构

### 模型配置（默认）

- **d_model**: 512 (模型维度)
- **n_layers**: 6 (编码器和解码器层数)
- **n_heads**: 8 (注意力头数)
- **d_ff**: 2048 (前馈网络维度)
- **dropout**: 0.1
- **max_src_len**: 512 (源序列最大长度)
- **max_tgt_len**: 128 (目标序列最大长度)

## 使用方法

### 训练模型

#### 方法1: 使用运行脚本（推荐）
```bash
bash scripts/run.sh
```

#### 方法2: 直接使用Python命令

完整训练命令（包含所有默认参数）：
```bash
python src/train.py \
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
    --seed 42 \
    --save_dir ./checkpoints \
    --results_dir ./results \
    --save_freq 5
```

#### 训练参数说明

**数据参数**:
- `--tokenizer`: Tokenizer名称（默认: bert-base-uncased）
- `--max_train_samples`: 训练集最大样本数（默认: 20000）
- `--max_val_samples`: 验证集最大样本数（默认: 5000）
- `--max_src_len`: 源序列最大长度（默认: 512）
- `--max_tgt_len`: 目标序列最大长度（默认: 128）
- `--batch_size`: Batch大小（默认: 16）
- `--num_workers`: 数据加载worker数量（默认: 4）

**模型参数**:
- `--d_model`: 模型维度（默认: 512）
- `--n_layers`: 编码器和解码器层数（默认: 6）
- `--n_heads`: 注意力头数（默认: 8）
- `--d_ff`: 前馈网络维度（默认: 2048）
- `--dropout`: Dropout率（默认: 0.1）

**训练参数**:
- `--epochs`: 训练轮数（默认: 10）
- `--learning_rate`: 学习率（默认: 1e-4）
- `--weight_decay`: 权重衰减（默认: 0.01）
- `--clip_grad`: 梯度裁剪阈值（默认: 1.0）
- `--scheduler`: 学习率调度器（cosine/plateau/none，默认: cosine）
- `--seed`: 随机种子（默认: 42）

**其他参数**:
- `--save_dir`: 模型保存目录（默认: ./checkpoints）
- `--results_dir`: 结果保存目录（默认: ./results）
- `--save_freq`: 保存检查点的频率（默认: 5）

### 评估模型

```bash
python src/evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --tokenizer bert-base-uncased \
    --max_val_samples 5000 \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --batch_size 16 \
    --num_workers 4 \
    --eval_samples 100 \
    --results_dir ./results
```

### 重现实验

使用以下精确命令可以重现实验结果（固定随机种子为42）：

```bash
# 设置随机种子
export SEED=42

# 训练
python src/train.py \
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
    --save_freq 5

# 评估
python src/evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --tokenizer bert-base-uncased \
    --max_val_samples 5000 \
    --max_src_len 512 \
    --max_tgt_len 128 \
    --batch_size 16 \
    --num_workers 4 \
    --eval_samples 100 \
    --results_dir ./results
```

## 训练特性

### 训练稳定性技巧

1. **AdamW优化器**: 使用带权重衰减的Adam优化器
2. **学习率调度**: 支持余弦退火（cosine）和基于验证损失的调度（plateau）
3. **梯度裁剪**: 防止梯度爆炸（默认阈值: 1.0）
4. **Dropout正则化**: 防止过拟合
5. **权重衰减**: L2正则化

### 训练监控

训练过程中会自动：
- 记录训练和验证损失
- 记录学习率变化
- 保存最佳模型（基于验证损失）
- 定期保存检查点
- 生成训练曲线图

训练完成后，`results/`目录会包含：
- `training_curves.png`: 训练和验证损失曲线、学习率曲线
- `training_history.json`: 训练历史记录

## 评估指标

模型评估使用以下指标：
- **BLEU**: 用于评估生成文本的质量
- **ROUGE-1**: 基于单词重叠的评估
- **ROUGE-2**: 基于双词重叠的评估
- **ROUGE-L**: 基于最长公共子序列的评估

评估结果保存在 `results/evaluation_results.json`。

## 模型文件说明

训练完成后，`checkpoints/`目录会包含：
- `best_model.pt`: 验证损失最低的模型
- `checkpoint_epoch_N.pt`: 定期保存的检查点
- `config.json`: 训练配置信息

## 代码结构说明

### `src/transformer.py`
包含Transformer模型的所有核心组件：
- `PositionalEncoding`: 位置编码
- `ScaledDotProductAttention`: 缩放点积注意力
- `MultiHeadAttention`: 多头注意力
- `PositionwiseFeedForward`: 位置前馈网络
- `EncoderLayer`: 编码器层
- `DecoderLayer`: 解码器层
- `Encoder`: 编码器
- `Decoder`: 解码器
- `Transformer`: 完整的Transformer模型

### `src/dataset.py`
数据集加载和预处理：
- `CNNDailyMailDataset`: CNN/DailyMail数据集类
- `load_cnn_dailymail_dataset`: 数据集加载函数

### `src/train.py`
训练脚本，包含：
- 模型训练循环
- 验证评估
- 学习率调度
- 模型保存和加载
- 训练曲线可视化

### `src/evaluate.py`
评估脚本，包含：
- 文本生成
- BLEU和ROUGE指标计算
- 模型性能评估

## 注意事项

1. **显存限制**: 如果显存不足，可以减小`batch_size`或`max_src_len`/`max_tgt_len`
2. **数据加载**: 首次运行会自动下载数据集，可能需要一些时间
3. **随机种子**: 使用固定随机种子（42）可以确保结果可重现
4. **依赖版本**: 建议使用requirements.txt中指定的版本范围

## 常见问题

### Q: 训练时显存不足怎么办？
A: 可以减小batch_size、max_src_len或max_tgt_len，或者使用梯度累积。

### Q: 如何调整模型大小？
A: 修改`--d_model`、`--n_layers`、`--n_heads`、`--d_ff`参数。

### Q: 训练速度慢怎么办？
A: 确保使用GPU训练，并检查`--num_workers`设置是否合理。

### Q: 如何继续训练？
A: 修改训练脚本加载checkpoint，并设置`--resume`参数（需要实现）。

## 参考文献

[1] Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems. 2017.

## 许可证

本项目仅供学习和研究使用。

## 作者

Student Name  
Student ID

