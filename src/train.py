"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import argparse
import os
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.dataset import load_cnn_dailymail_dataset


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, train_loader, criterion, optimizer, device, pad_idx, clip_grad=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        # 生成mask
        src_mask, tgt_mask = model.generate_mask(src, tgt_input, pad_idx)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算loss（只对非padding位置）
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        loss = criterion(output, tgt_output)
        
        # 只计算非padding的token
        non_pad_mask = (tgt_output != pad_idx)
        loss = loss * non_pad_mask.float()
        loss = loss.sum() / non_pad_mask.sum().float()
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_tokens += non_pad_mask.sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader), total_tokens


def evaluate(model, val_loader, criterion, device, pad_idx):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # 生成mask
            src_mask, tgt_mask = model.generate_mask(src, tgt_input, pad_idx)
            
            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算loss
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            
            # 只计算非padding的token
            non_pad_mask = (tgt_output != pad_idx)
            loss = loss * non_pad_mask.float()
            loss = loss.sum() / non_pad_mask.sum().float()
            
            total_loss += loss.item()
            total_tokens += non_pad_mask.sum().item()
    
    return total_loss / len(val_loader), total_tokens


def train(args):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据集...")
    train_loader, val_loader, tokenizer = load_cnn_dailymail_dataset(
        tokenizer_name=args.tokenizer,
        max_samples_train=args.max_train_samples,
        max_samples_val=args.max_val_samples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_dir=args.dataset_dir
    )
    
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        dropout=args.dropout
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"模型参数量: {num_params:,}")
    
    # Loss和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['num_params'] = num_params
    config['vocab_size'] = vocab_size
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_tokens = train_epoch(
            model, train_loader, criterion, optimizer, device, pad_idx, args.clip_grad
        )
        
        # 验证
        val_loss, val_tokens = evaluate(
            model, val_loader, criterion, device, pad_idx
        )
        
        # 学习率调度
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 定期保存检查点
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'training_curves.png'))
    plt.close()
    
    # 保存历史
    with open(os.path.join(args.results_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    
    # 数据参数
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Tokenizer名称')
    parser.add_argument('--max_train_samples', type=int, default=20000,
                        help='训练集最大样本数')
    parser.add_argument('--max_val_samples', type=int, default=5000,
                        help='验证集最大样本数')
    parser.add_argument('--max_src_len', type=int, default=512,
                        help='源序列最大长度')
    parser.add_argument('--max_tgt_len', type=int, default=128,
                        help='目标序列最大长度')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载worker数量')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='本地数据集目录路径')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='层数')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='FFN维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='学习率调度器')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='保存检查点的频率')
    
    args = parser.parse_args()
    
    train(args)

