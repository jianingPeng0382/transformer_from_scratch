"""
下载CNN/DailyMail数据集到本地dataset文件夹
"""
import os
from datasets import load_dataset
from huggingface_hub import login

def download_dataset(token=None):
    """下载CNN/DailyMail数据集到本地"""
    dataset_dir = './dataset'
    
    # 创建dataset目录
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 如果提供了token，先登录Hugging Face
    if token:
        print("正在使用提供的token登录Hugging Face...")
        try:
            login(token=token)
            print("登录成功！")
        except Exception as e:
            print(f"登录失败: {e}")
            print("尝试继续下载（可能会失败）...")
    else:
        print("未提供token，尝试不使用认证下载...")
    
    print("\n正在从Hugging Face下载CNN/DailyMail数据集...")
    print("这可能需要一些时间，请耐心等待...")
    
    # 加载数据集（如果提供了token，也在这里传递）
    if token:
        dataset = load_dataset('cnn_dailymail', '3.0.0', token=token)
    else:
        dataset = load_dataset('cnn_dailymail', '3.0.0')
    
    print(f"数据集已加载")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['validation'])}")
    print(f"测试集样本数: {len(dataset['test'])}")
    
    # 保存到本地
    print(f"\n正在保存数据集到 {dataset_dir} ...")
    dataset.save_to_disk(dataset_dir)
    
    print(f"\n数据集已成功保存到 {dataset_dir}/")
    print("现在可以使用本地数据集了！")

if __name__ == '__main__':
    # Hugging Face token
    hf_token = "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"
    download_dataset(token=hf_token)

