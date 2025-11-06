"""
CNN/DailyMail数据集加载和预处理
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os


class CNNDailyMailDataset(Dataset):
    """CNN/DailyMail数据集"""
    
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_token_id = tokenizer.pad_token_id
        
        # 处理EOS token - 如果没有则使用SEP token
        if tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
            self.eos_token_id = tokenizer.sep_token_id
        else:
            # 如果都没有，使用CLS token作为替代
            self.eos_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None else 102
        
        # 处理BOS token - 如果没有则使用CLS token
        if tokenizer.bos_token_id is not None:
            self.bos_token_id = tokenizer.bos_token_id
        elif hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
            self.bos_token_id = tokenizer.cls_token_id
        else:
            # 如果都没有，使用EOS token或默认值
            self.bos_token_id = self.eos_token_id
        
        # 确保所有token ID都是整数类型（防止None值）
        try:
            self.pad_token_id = int(self.pad_token_id) if self.pad_token_id is not None else 0
        except (TypeError, ValueError):
            self.pad_token_id = 0
        
        try:
            self.eos_token_id = int(self.eos_token_id) if self.eos_token_id is not None else 102
        except (TypeError, ValueError):
            self.eos_token_id = 102
        
        try:
            self.bos_token_id = int(self.bos_token_id) if self.bos_token_id is not None else 101
        except (TypeError, ValueError):
            self.bos_token_id = 101
        
        # 最终检查，确保不是None
        assert self.pad_token_id is not None, "pad_token_id should not be None"
        assert self.eos_token_id is not None, "eos_token_id should not be None"
        assert self.bos_token_id is not None, "bos_token_id should not be None"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取article和highlights
        article = item['article']
        highlights = item['highlights']
        
        # Tokenize
        src_tokens = self.tokenizer(
            article,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # 对于target，添加BOS和EOS token
        tgt_tokens = self.tokenizer(
            highlights,
            max_length=self.max_tgt_len - 2,  # 预留BOS和EOS位置
            padding=False,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        # 添加BOS和EOS（确保token ID是整数）
        tgt_tokens = torch.cat([
            torch.tensor([self.bos_token_id], dtype=torch.long),
            tgt_tokens,
            torch.tensor([self.eos_token_id], dtype=torch.long)
        ])
        
        # Padding target
        if len(tgt_tokens) < self.max_tgt_len:
            padding = torch.full((self.max_tgt_len - len(tgt_tokens),), self.pad_token_id)
            tgt_tokens = torch.cat([tgt_tokens, padding])
        else:
            tgt_tokens = tgt_tokens[:self.max_tgt_len]
        
        # 创建输入和目标序列
        # 对于训练，输入是target的前n-1个token，目标是target的后n-1个token
        tgt_input = tgt_tokens[:-1]
        tgt_output = tgt_tokens[1:]
        
        return {
            'src': src_tokens,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'src_len': (src_tokens != self.pad_token_id).sum().item(),
            'tgt_len': (tgt_tokens != self.pad_token_id).sum().item()
        }


def load_cnn_dailymail_dataset(tokenizer_name='bert-base-uncased', 
                                 max_samples_train=20000, 
                                 max_samples_val=5000,
                                 max_src_len=512,
                                 max_tgt_len=128,
                                 batch_size=32,
                                 num_workers=4,
                                 dataset_dir='./dataset'):
    """
    加载CNN/DailyMail数据集
    
    Args:
        tokenizer_name: tokenizer名称
        max_samples_train: 训练集最大样本数
        max_samples_val: 验证集最大样本数
        max_src_len: 源序列最大长度
        max_tgt_len: 目标序列最大长度
        batch_size: batch大小
        num_workers: 数据加载的worker数量
        dataset_dir: 本地数据集目录路径，如果目录存在则从本地加载，否则从Hugging Face下载
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 设置特殊token（对于BERT等没有这些token的tokenizer）
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 对于BERT，通常没有BOS和EOS token，使用CLS和SEP
    if tokenizer.bos_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.bos_token = tokenizer.eos_token
        elif hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
            tokenizer.bos_token = tokenizer.cls_token
    
    if tokenizer.eos_token is None:
        if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        elif hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
            tokenizer.eos_token = tokenizer.cls_token
    
    # 加载数据集（优先从本地加载）
    print("正在加载CNN/DailyMail数据集...")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir) and os.listdir(dataset_dir):
        print(f"从本地目录加载数据集: {dataset_dir}")
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_dir)
            print("成功从本地加载数据集")
        except Exception as e:
            print(f"从本地加载失败: {e}")
            print("尝试从Hugging Face下载...")
            dataset = load_dataset('cnn_dailymail', '3.0.0')
    else:
        if os.path.exists(dataset_dir):
            print(f"本地数据集目录存在但为空: {dataset_dir}")
        else:
            print(f"本地数据集目录不存在: {dataset_dir}")
        print("从Hugging Face下载数据集...")
        dataset = load_dataset('cnn_dailymail', '3.0.0')
    
    # 选择子集（如果指定）
    train_data = dataset['train']
    val_data = dataset['validation']
    
    if max_samples_train > 0:
        train_data = train_data.select(range(min(max_samples_train, len(train_data))))
    if max_samples_val > 0:
        val_data = val_data.select(range(min(max_samples_val, len(val_data))))
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    
    # 创建Dataset
    train_dataset = CNNDailyMailDataset(train_data, tokenizer, max_src_len, max_tgt_len)
    val_dataset = CNNDailyMailDataset(val_data, tokenizer, max_src_len, max_tgt_len)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

