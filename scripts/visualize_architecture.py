"""
绘制Transformer架构图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_transformer_architecture():
    """绘制Transformer架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'encoder': '#E8F4F8',
        'decoder': '#FFF4E6',
        'attention': '#FFE5E5',
        'ffn': '#E5F5E5',
        'embed': '#F0E5FF',
        'output': '#FFE5F0'
    }
    
    # 绘制Encoder部分
    # Encoder Input Embedding
    encoder_embed = FancyBboxPatch((0.5, 9.5), 2, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['embed'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(encoder_embed)
    ax.text(1.5, 9.9, 'Input\nEmbedding', ha='center', va='center', fontsize=10, weight='bold')
    
    # Encoder Positional Encoding
    encoder_pos = FancyBboxPatch((0.5, 8.3), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['embed'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(encoder_pos)
    ax.text(1.5, 8.7, 'Positional\nEncoding', ha='center', va='center', fontsize=10, weight='bold')
    
    # Encoder Stack (N layers)
    for i in range(3):  # 显示3层作为示例
        y_pos = 6.5 - i * 1.5
        
        # Encoder Layer框
        encoder_layer = FancyBboxPatch((0.3, y_pos - 0.6), 2.4, 1.2, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor=colors['encoder'], 
                                       edgecolor='black', linewidth=2)
        ax.add_patch(encoder_layer)
        
        # Multi-Head Attention
        mha = FancyBboxPatch((0.5, y_pos - 0.3), 1, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['attention'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(mha)
        ax.text(1, y_pos - 0.1, 'Multi-Head\nAttention', ha='center', va='center', fontsize=8)
        
        # Add & Norm
        add_norm1 = FancyBboxPatch((1.7, y_pos - 0.3), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFD700', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(add_norm1)
        ax.text(2, y_pos - 0.1, 'Add &\nNorm', ha='center', va='center', fontsize=7)
        
        # Feed Forward
        ffn = FancyBboxPatch((0.5, y_pos - 0.5), 1, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['ffn'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(ffn)
        ax.text(1, y_pos - 0.3, 'Feed\nForward', ha='center', va='center', fontsize=8)
        
        # Add & Norm 2
        add_norm2 = FancyBboxPatch((1.7, y_pos - 0.5), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFD700', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(add_norm2)
        ax.text(2, y_pos - 0.3, 'Add &\nNorm', ha='center', va='center', fontsize=7)
        
        # 残差连接箭头
        if i > 0:
            # 从上层到当前层的残差连接
            arrow1 = FancyArrowPatch((2.5, 6.5 - (i-1) * 1.5), 
                                     (2.5, y_pos + 0.3),
                                     arrowstyle='->', mutation_scale=20, 
                                     linewidth=1.5, color='blue', linestyle='--')
            ax.add_patch(arrow1)
    
    ax.text(1.5, 2.5, 'N×', ha='center', va='center', fontsize=14, weight='bold')
    
    # Encoder Output
    encoder_output = FancyBboxPatch((0.5, 1.5), 2, 0.6, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['encoder'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(encoder_output)
    ax.text(1.5, 1.8, 'Encoder Output', ha='center', va='center', fontsize=10, weight='bold')
    
    # 绘制Decoder部分
    # Decoder Input Embedding
    decoder_embed = FancyBboxPatch((6, 9.5), 2, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['embed'], 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(decoder_embed)
    ax.text(7, 9.9, 'Output\nEmbedding', ha='center', va='center', fontsize=10, weight='bold')
    
    # Decoder Positional Encoding
    decoder_pos = FancyBboxPatch((6, 8.3), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['embed'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(decoder_pos)
    ax.text(7, 8.7, 'Positional\nEncoding', ha='center', va='center', fontsize=10, weight='bold')
    
    # Decoder Stack (N layers)
    for i in range(3):  # 显示3层作为示例
        y_pos = 6.5 - i * 1.5
        
        # Decoder Layer框
        decoder_layer = FancyBboxPatch((5.8, y_pos - 0.6), 2.4, 1.2, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor=colors['decoder'], 
                                       edgecolor='black', linewidth=2)
        ax.add_patch(decoder_layer)
        
        # Masked Multi-Head Attention
        mmha = FancyBboxPatch((6, y_pos - 0.1), 1, 0.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['attention'], 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(mmha)
        ax.text(6.5, y_pos + 0.1, 'Masked\nMulti-Head\nAttention', ha='center', va='center', fontsize=7)
        
        # Add & Norm
        add_norm1 = FancyBboxPatch((7.2, y_pos - 0.1), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFD700', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(add_norm1)
        ax.text(7.5, y_pos + 0.1, 'Add &\nNorm', ha='center', va='center', fontsize=7)
        
        # Multi-Head Attention (Cross-attention)
        cross_attn = FancyBboxPatch((6, y_pos - 0.4), 1, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFB6C1', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(cross_attn)
        ax.text(6.5, y_pos - 0.2, 'Multi-Head\nAttention', ha='center', va='center', fontsize=7)
        
        # Add & Norm
        add_norm2 = FancyBboxPatch((7.2, y_pos - 0.4), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFD700', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(add_norm2)
        ax.text(7.5, y_pos - 0.2, 'Add &\nNorm', ha='center', va='center', fontsize=7)
        
        # Feed Forward
        ffn = FancyBboxPatch((6, y_pos - 0.6), 1, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['ffn'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(ffn)
        ax.text(6.5, y_pos - 0.4, 'Feed\nForward', ha='center', va='center', fontsize=8)
        
        # Add & Norm 3
        add_norm3 = FancyBboxPatch((7.2, y_pos - 0.6), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='#FFD700', 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(add_norm3)
        ax.text(7.5, y_pos - 0.4, 'Add &\nNorm', ha='center', va='center', fontsize=7)
    
    ax.text(7, 2.5, 'N×', ha='center', va='center', fontsize=14, weight='bold')
    
    # Decoder Output
    decoder_output = FancyBboxPatch((6, 1.5), 2, 0.6, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['decoder'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(decoder_output)
    ax.text(7, 1.8, 'Decoder Output', ha='center', va='center', fontsize=10, weight='bold')
    
    # Linear & Softmax
    linear = FancyBboxPatch((6, 0.5), 2, 0.6, 
                           boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], 
                           edgecolor='black', linewidth=2)
    ax.add_patch(linear)
    ax.text(7, 0.8, 'Linear & Softmax', ha='center', va='center', fontsize=10, weight='bold')
    
    # 连接线 - Encoder到Decoder
    for i in range(3):
        y_pos = 6.5 - i * 1.5
        arrow = FancyArrowPatch((2.7, y_pos - 0.1), 
                               (5.8, y_pos - 0.1),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='red')
        ax.add_patch(arrow)
    
    # 垂直连接线
    # Encoder
    for i in range(4):
        y_start = 9.3 - i * 0.8
        y_end = 9.1 - i * 0.8
        if y_start > 2.5:
            line = plt.Line2D([1.5, 1.5], [y_start, y_end], 
                             linewidth=2, color='black')
            ax.add_line(line)
    
    # Decoder
    for i in range(4):
        y_start = 9.3 - i * 0.8
        y_end = 9.1 - i * 0.8
        if y_start > 2.5:
            line = plt.Line2D([7, 7], [y_start, y_end], 
                             linewidth=2, color='black')
            ax.add_line(line)
    
    # 标题
    ax.text(5, 11.5, 'Transformer Architecture', ha='center', va='center', 
           fontsize=18, weight='bold')
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['encoder'], edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=colors['decoder'], edgecolor='black', label='Decoder'),
        mpatches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention'),
        mpatches.Patch(facecolor=colors['ffn'], edgecolor='black', label='Feed Forward'),
        mpatches.Patch(facecolor=colors['embed'], edgecolor='black', label='Embedding'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Add & Norm'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("正在生成Transformer架构图...")
    fig = draw_transformer_architecture()
    fig.savefig('figures/transformer_architecture.png', dpi=300, bbox_inches='tight')
    print("架构图已保存到: figures/transformer_architecture.png")
    plt.close()

