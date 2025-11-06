"""
Transformer模型实现
包含Multi-Head Attention、FFN、Encoder、Decoder等核心组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码（正弦/余弦）"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, n_heads, seq_len_q, d_k]
            K: [batch_size, n_heads, seq_len_k, d_k]
            V: [batch_size, n_heads, seq_len_v, d_v]
            mask: [batch_size, 1, seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
        Returns:
            output: [batch_size, n_heads, seq_len_q, d_v]
            attn_weights: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask
        # mask中True表示不mask（有效位置），False表示mask（无效位置）
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            # 如果mask是bool类型，False的位置需要mask掉
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -1e9)
            else:
                # 如果mask是数值类型，0的位置需要mask掉
                scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, query_len, d_model]
            key: [batch_size, key_len, d_model]
            value: [batch_size, value_len, d_model]
            mask: [batch_size, query_len, key_len] 或 [batch_size, 1, query_len, key_len]
        Returns:
            output: [batch_size, query_len, d_model]
        """
        batch_size, query_len = query.size(0), query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)
        
        # 残差连接
        residual = query
        
        # 线性投影并重塑为多头
        Q = self.W_q(query).view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, value_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络（FFN）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x
        output = self.linear1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output


class EncoderLayer(nn.Module):
    """Encoder层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Self-attention
        x = self.self_attn(x, x, x, mask)
        
        # Feed-forward
        x = self.feed_forward(x)
        
        return x


class DecoderLayer(nn.Module):
    """Decoder层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # Self-attention (with future mask)
        x = self.self_attn(x, x, x, tgt_mask)
        
        # Cross-attention
        # 扩展src_mask用于cross-attention: [batch_size, tgt_seq_len, src_seq_len]
        # 对于cross-attention，query来自decoder，key/value来自encoder
        src_mask_expanded = src_mask.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch_size, tgt_seq_len, src_seq_len]
        x = self.cross_attn(x, encoder_output, encoder_output, src_mask_expanded)
        
        # Feed-forward
        x = self.feed_forward(x)
        
        return x


class Encoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        """
        Args:
            src: [batch_size, src_seq_len]
            src_mask: [batch_size, src_seq_len]
        Returns:
            [batch_size, src_seq_len, d_model]
        """
        # Embedding
        x = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # 转换为 [seq_len, batch_size, d_model] 用于位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # 转回 [batch_size, seq_len, d_model]
        
        x = self.dropout(x)
        
        # 扩展mask维度
        src_mask_expanded = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        src_mask_expanded = src_mask_expanded.expand(-1, -1, src_mask.size(1), -1)
        
        # Encoder layers
        for layer in self.layers:
            x = layer(x, src_mask_expanded)
        
        return x


class Decoder(nn.Module):
    """Transformer Decoder"""
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        Args:
            tgt: [batch_size, tgt_seq_len]
            encoder_output: [batch_size, src_seq_len, d_model]
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        Returns:
            [batch_size, tgt_seq_len, d_model]
        """
        # Embedding
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # 转换为 [seq_len, batch_size, d_model] 用于位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # 转回 [batch_size, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型（Encoder-Decoder架构）"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, max_src_len=5000, max_tgt_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_src_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_tgt_len, dropout)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len]
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        Returns:
            [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate_mask(self, src, tgt, pad_idx=0):
        """
        生成mask
        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len]
            pad_idx: padding token的索引
        Returns:
            src_mask: [batch_size, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        """
        # Source mask (padding mask) - 使用bool类型
        src_mask = (src != pad_idx).bool()
        
        # Target mask (padding mask + future mask)
        tgt_pad_mask = (tgt != pad_idx).bool()  # [batch_size, tgt_seq_len]
        tgt_seq_len = tgt.size(1)
        
        # 创建future mask (下三角矩阵) - 使用bool类型
        tgt_future_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
        tgt_future_mask = tgt_future_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
        
        # 组合padding mask和future mask（都是bool类型）
        tgt_mask = tgt_pad_mask.unsqueeze(1) & tgt_future_mask  # [batch_size, tgt_seq_len, tgt_seq_len]
        
        return src_mask, tgt_mask

