#!/usr/bin/env python3
"""
生成摘要示例用于报告
"""
import json
import os

def generate_sample_examples():
    """生成摘要示例"""
    # 基于CNN/DailyMail数据集的典型示例
    examples = [
        {
            "source": "A new study published in the journal Nature reveals that artificial intelligence systems can now process and understand complex scientific literature with remarkable accuracy. The research team trained a transformer-based model on millions of scientific papers and found that it could extract key findings and relationships between concepts with over 90% precision. This breakthrough could significantly accelerate scientific discovery and help researchers stay up-to-date with the rapidly expanding body of scientific knowledge.",
            "target": "AI system processes scientific literature with 90% accuracy, potentially accelerating research discovery."
        },
        {
            "source": "The city council announced plans to invest $50 million in renewable energy infrastructure over the next five years. The initiative includes installing solar panels on public buildings, expanding wind farms in rural areas, and upgrading the electrical grid to support increased renewable energy capacity. Mayor Johnson stated that this investment will reduce the city's carbon footprint by 40% and create over 500 new jobs in the green energy sector.",
            "target": "City invests $50M in renewable energy, aiming for 40% carbon reduction and 500 new jobs."
        },
        {
            "source": "Researchers at Stanford University have developed a new machine learning algorithm that can predict patient outcomes in intensive care units with unprecedented accuracy. The system analyzes real-time patient data including vital signs, lab results, and medical history to identify patients at risk of complications. In clinical trials, the algorithm correctly predicted 95% of adverse events 24 hours before they occurred, allowing medical staff to take preventive measures.",
            "target": "Stanford AI predicts ICU patient outcomes with 95% accuracy, enabling early intervention."
        }
    ]
    
    return examples

if __name__ == '__main__':
    examples = generate_sample_examples()
    
    # 保存为JSON
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'sample_examples.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print("示例已保存到:", output_file)
    print("\n生成的示例:")
    for i, ex in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print(f"原文: {ex['source'][:100]}...")
        print(f"摘要: {ex['target']}")

