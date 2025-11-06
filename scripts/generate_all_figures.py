#!/usr/bin/env python3
"""
生成所有报告所需的图表
"""
import os
import sys
import subprocess

def main():
    """生成所有图表"""
    print("=" * 70)
    print("生成所有报告图表")
    print("=" * 70)
    
    # 确保目录存在
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. 生成Transformer架构图
    print("\n[1/2] 生成Transformer架构图...")
    try:
        subprocess.run([sys.executable, 'scripts/visualize_architecture.py'], 
                      check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print("  ✓ Transformer架构图已生成")
    except Exception as e:
        print(f"  ✗ 生成架构图失败: {e}")
    
    # 2. 生成消融实验图表
    print("\n[2/2] 生成消融实验图表...")
    try:
        subprocess.run([sys.executable, 'scripts/generate_ablation_plots.py'], 
                      check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print("  ✓ 消融实验图表已生成")
    except Exception as e:
        print(f"  ✗ 生成消融实验图表失败: {e}")
    
    # 验证文件
    print("\n" + "=" * 70)
    print("验证生成的文件...")
    print("=" * 70)
    
    required_files = [
        'figures/transformer_architecture.png',
        'figures/ablation_positional_encoding.png',
        'figures/ablation_attention_heads.png',
        'figures/ablation_model_depth.png',
        'results/training_curves.png'
    ]
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    all_exist = True
    
    for file in required_files:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file} (缺失)")
            all_exist = False
    
    print("\n" + "=" * 70)
    if all_exist:
        print("✓ 所有图表已成功生成！")
    else:
        print("✗ 部分图表缺失，请检查错误信息")
    print("=" * 70)

if __name__ == '__main__':
    main()

