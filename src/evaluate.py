"""
评估和生成脚本
"""
import torch
import argparse
import json
import os
from tqdm import tqdm
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("Warning: nltk not installed. BLEU scores will not be calculated.")
    sentence_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Warning: rouge-score not installed. ROUGE scores will not be calculated.")
    rouge_scorer = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.dataset import load_cnn_dailymail_dataset


def generate(model, src, src_mask, tokenizer, max_len=128, beam_size=1, device='cuda'):
    """
    生成摘要
    
    Args:
        model: Transformer模型
        src: 源序列 [batch_size, src_len]
        src_mask: 源mask [batch_size, src_len]
        tokenizer: tokenizer
        max_len: 最大生成长度
        beam_size: beam search大小（1表示贪心搜索）
        device: 设备
    
    Returns:
        生成的token序列
    """
    model.eval()
    batch_size = src.size(0)
    pad_idx = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    
    # 处理BOS token ID
    if tokenizer.bos_token_id is not None:
        bos_idx = int(tokenizer.bos_token_id)
    elif hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        bos_idx = int(tokenizer.cls_token_id)
    elif tokenizer.eos_token_id is not None:
        bos_idx = int(tokenizer.eos_token_id)
    else:
        bos_idx = 101  # BERT CLS token default
    
    # 处理EOS token ID
    if tokenizer.eos_token_id is not None:
        eos_idx = int(tokenizer.eos_token_id)
    elif hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        eos_idx = int(tokenizer.sep_token_id)
    elif hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        eos_idx = int(tokenizer.cls_token_id)
    else:
        eos_idx = 102  # BERT SEP token default
    
    # Encoder前向传播
    encoder_output = model.encoder(src, src_mask)
    
    # 初始化decoder输入
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            # 生成tgt mask
            _, tgt_mask = model.generate_mask(src, tgt, pad_idx)
            
            # Decoder前向传播
            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            
            # 输出投影
            output = model.output_projection(decoder_output)
            
            # 获取下一个token
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 拼接
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都生成结束
            if (next_token == eos_idx).all():
                break
    
    return tgt


def calculate_bleu(pred_tokens, target_tokens, tokenizer):
    """计算BLEU分数"""
    if sentence_bleu is None:
        return 0.0
    
    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
    
    pred_words = pred_text.split()
    target_words = target_text.split()
    
    if len(target_words) == 0:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    return sentence_bleu([target_words], pred_words, smoothing_function=smoothing)


def evaluate_model(model, val_loader, tokenizer, device, max_samples=100):
    """评估模型"""
    model.eval()
    
    bleu_scores = []
    rouge_scores = []
    
    if rouge_scorer is not None:
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    else:
        rouge_scorer_obj = None
    
    pad_idx = tokenizer.pad_token_id
    
    print("开始评估...")
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if count >= max_samples:
                break
            
            src = batch['src'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # 生成mask
            src_mask, _ = model.generate_mask(src, src, pad_idx)
            src_mask = (src != pad_idx).long()
            
            # 生成
            generated = generate(model, src, src_mask, tokenizer, max_len=128, device=device)
            
            # 计算指标
            for i in range(src.size(0)):
                if count >= max_samples:
                    break
                
                # BLEU
                pred_tokens = generated[i].cpu().numpy()
                target_tokens = tgt_output[i].cpu().numpy()
                
                # 移除padding和特殊token
                # 获取实际的特殊token ID
                pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
                bos_id = int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else (int(tokenizer.cls_token_id) if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None else 101)
                eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else (int(tokenizer.sep_token_id) if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None else 102)
                
                pred_tokens = [t for t in pred_tokens if t != pad_id and t != bos_id and t != eos_id]
                target_tokens = [t for t in target_tokens if t != pad_id and t != bos_id and t != eos_id]
                
                bleu = calculate_bleu(pred_tokens, target_tokens, tokenizer)
                bleu_scores.append(bleu)
                
                # ROUGE
                if rouge_scorer_obj is not None:
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
                    
                    rouge = rouge_scorer_obj.score(target_text, pred_text)
                    rouge_scores.append(rouge)
                
                count += 1
    
    # 计算平均分数
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_rouge1 = np.mean([r['rouge1'].fmeasure for r in rouge_scores]) if rouge_scores else 0.0
    avg_rouge2 = np.mean([r['rouge2'].fmeasure for r in rouge_scores]) if rouge_scores else 0.0
    avg_rougel = np.mean([r['rougeL'].fmeasure for r in rouge_scores]) if rouge_scores else 0.0
    
    results = {
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougel,
        'num_samples': count
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估Transformer模型')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Tokenizer名称')
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
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='评估样本数')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载检查点
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # 加载数据
    print("加载数据集...")
    _, val_loader, tokenizer = load_cnn_dailymail_dataset(
        tokenizer_name=args.tokenizer,
        max_samples_train=0,
        max_samples_val=args.max_val_samples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_dir=args.dataset_dir
    )
    
    # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=config['vocab_size'],
        tgt_vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_src_len=config['max_src_len'],
        max_tgt_len=config['max_tgt_len'],
        dropout=config['dropout']
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    results = evaluate_model(model, val_loader, tokenizer, device, args.eval_samples)
    
    # 打印结果
    print("\n评估结果:")
    print(f"BLEU: {results['bleu']:.4f}")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    
    # 保存结果
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 {args.results_dir}/evaluation_results.json")


if __name__ == '__main__':
    main()

