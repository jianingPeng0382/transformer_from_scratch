"""
运行消融实验并生成图表
"""
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train
from src.transformer import Transformer
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_ablation_experiment(experiment_type, base_args, output_dir='./ablation_results'):
    """运行消融实验"""
    os.makedirs(output_dir, exist_ok=True)
    
    if experiment_type == 'positional_encoding':
        # 位置编码消融实验
        configs = [
            {'name': 'with_positional_encoding', 'modify_model': None},
            {'name': 'without_positional_encoding', 'modify_model': 'remove_pos_encoding'}
        ]
    elif experiment_type == 'attention_heads':
        # 注意力头数消融实验
        configs = [
            {'name': 'heads_2', 'n_heads': 2},
            {'name': 'heads_4', 'n_heads': 4},
            {'name': 'heads_8', 'n_heads': 8},
            {'name': 'heads_16', 'n_heads': 16}
        ]
    elif experiment_type == 'model_depth':
        # 模型深度消融实验
        configs = [
            {'name': 'layers_2', 'n_layers': 2},
            {'name': 'layers_4', 'n_layers': 4},
            {'name': 'layers_6', 'n_layers': 6},
            {'name': 'layers_8', 'n_layers': 8}
        ]
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*60}")
        
        # 创建实验参数
        exp_args = argparse.Namespace(**vars(base_args))
        
        # 更新参数
        for key, value in config.items():
            if key != 'name' and key != 'modify_model':
                setattr(exp_args, key, value)
        
        # 设置实验特定的保存目录
        exp_args.save_dir = os.path.join(output_dir, experiment_type, config['name'])
        exp_args.results_dir = os.path.join(output_dir, experiment_type, config['name'], 'results')
        os.makedirs(exp_args.save_dir, exist_ok=True)
        os.makedirs(exp_args.results_dir, exist_ok=True)
        
        # 运行训练
        if config.get('modify_model') == 'remove_pos_encoding':
            # 需要修改模型以移除位置编码
            # 这里我们通过设置一个标志来在训练时处理
            train_without_pos_encoding(exp_args)
        else:
            train(exp_args)
        
        # 加载训练历史
        history_file = os.path.join(exp_args.results_dir, 'training_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                results[config['name']] = json.load(f)
        else:
            print(f"Warning: History file not found: {history_file}")
    
    return results

def train_without_pos_encoding(args):
    """训练没有位置编码的模型"""
    # 导入训练相关函数
    from src.train import train_epoch, evaluate, count_parameters
    from src.dataset import load_cnn_dailymail_dataset
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm
    import numpy as np
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    
    # 创建模型（不使用位置编码）
    from src.transformer import Encoder, Decoder
    
    # 创建自定义模型，不使用位置编码
    class TransformerNoPosEncoding(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                     n_heads=8, d_ff=2048, max_src_len=5000, max_tgt_len=5000, dropout=0.1):
            super().__init__()
            
            # Encoder without positional encoding
            self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
            self.encoder_layers = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.encoder_dropout = nn.Dropout(dropout)
            
            # Decoder without positional encoding
            self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.decoder_dropout = nn.Dropout(dropout)
            
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
            self.d_model = d_model
        
        def forward(self, src, tgt, src_mask, tgt_mask):
            # Encoder
            x = self.encoder_embedding(src) * np.sqrt(self.d_model)
            x = self.encoder_dropout(x)
            src_mask_expanded = src_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, src_mask.size(1), -1)
            for layer in self.encoder_layers:
                x = layer(x, src_mask_expanded)
            encoder_output = x
            
            # Decoder
            x = self.decoder_embedding(tgt) * np.sqrt(self.d_model)
            x = self.decoder_dropout(x)
            for layer in self.decoder_layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
            
            output = self.output_projection(x)
            return output
        
        def generate_mask(self, src, tgt, pad_idx=0):
            src_mask = (src != pad_idx).bool()
            tgt_pad_mask = (tgt != pad_idx).bool()
            tgt_seq_len = tgt.size(1)
            tgt_future_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
            tgt_future_mask = tgt_future_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
            tgt_mask = tgt_pad_mask.unsqueeze(1) & tgt_future_mask
            return src_mask, tgt_mask
    
    from src.transformer import EncoderLayer, DecoderLayer
    
    model = TransformerNoPosEncoding(
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
    
    # 继续使用原来的训练循环
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    print("\n开始训练（无位置编码）...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        model.train()
        train_loss_list = []
        for batch in tqdm(train_loader, desc='Training'):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            src_mask, tgt_mask = model.generate_mask(src, tgt_input, pad_idx)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.view(-1, output.size(-1))
            tgt_output_flat = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output_flat)
            non_pad_mask = (tgt_output_flat != pad_idx)
            loss = (loss * non_pad_mask.float()).sum() / non_pad_mask.sum().float()
            
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            train_loss_list.append(loss.item())
        
        train_loss = np.mean(train_loss_list)
        
        # 验证
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                src = batch['src'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)
                
                src_mask, tgt_mask = model.generate_mask(src, tgt_input, pad_idx)
                output = model(src, tgt_input, src_mask, tgt_mask)
                
                output = output.view(-1, output.size(-1))
                tgt_output_flat = tgt_output.view(-1)
                
                loss = criterion(output, tgt_output_flat)
                non_pad_mask = (tgt_output_flat != pad_idx)
                loss = (loss * non_pad_mask.float()).sum() / non_pad_mask.sum().float()
                val_loss_list.append(loss.item())
        
        val_loss = np.mean(val_loss_list)
        
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # 保存历史
    with open(os.path.join(args.results_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

def plot_ablation_results(results, experiment_type, output_path):
    """绘制消融实验结果"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if experiment_type == 'positional_encoding':
        # 位置编码对比
        if 'with_positional_encoding' in results and 'without_positional_encoding' in results:
            epochs = range(1, len(results['with_positional_encoding']['val_loss']) + 1)
            ax.plot(epochs, results['with_positional_encoding']['val_loss'], 
                   'o-', label='With Positional Encoding', linewidth=2, markersize=6)
            ax.plot(epochs, results['without_positional_encoding']['val_loss'], 
                   's-', label='Without Positional Encoding', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Validation Loss', fontsize=12)
            ax.set_title('Effect of Positional Encoding', fontsize=14, weight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
    
    elif experiment_type == 'attention_heads':
        # 注意力头数对比
        for name, data in results.items():
            if 'val_loss' in data:
                epochs = range(1, len(data['val_loss']) + 1)
                label = name.replace('heads_', '').replace('_', ' ') + ' heads'
                ax.plot(epochs, data['val_loss'], 'o-', label=label, linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Effect of Number of Attention Heads', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    elif experiment_type == 'model_depth':
        # 模型深度对比
        for name, data in results.items():
            if 'val_loss' in data:
                epochs = range(1, len(data['val_loss']) + 1)
                label = name.replace('layers_', '').replace('_', ' ') + ' layers'
                ax.plot(epochs, data['val_loss'], 'o-', label=label, linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Effect of Model Depth (Number of Layers)', fontsize=14, weight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行消融实验')
    
    # 实验类型
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['positional_encoding', 'attention_heads', 'model_depth', 'all'],
                       help='实验类型')
    
    # 数据参数
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--max_train_samples', type=int, default=5000)  # 消融实验使用较少数据
    parser.add_argument('--max_val_samples', type=int, default=1000)
    parser.add_argument('--max_src_len', type=int, default=512)
    parser.add_argument('--max_tgt_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    
    # 模型参数（默认值）
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5)  # 消融实验使用较少epoch
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('figures', exist_ok=True)
    
    experiments = ['positional_encoding', 'attention_heads', 'model_depth'] if args.experiment == 'all' else [args.experiment]
    
    for exp_type in experiments:
        print(f"\n{'='*70}")
        print(f"开始运行消融实验: {exp_type}")
        print(f"{'='*70}")
        
        results = run_ablation_experiment(exp_type, args)
        
        # 绘制结果
        output_path = f'figures/ablation_{exp_type}.png'
        plot_ablation_results(results, exp_type, output_path)

