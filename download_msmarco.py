import os
from datasets import load_dataset

# --- 配置 ---

# 1. 定义输出文件夹和文件名
output_folder = "data"
# 我们换个文件名，以反映它的大小
output_filename = os.path.join(output_folder, "msmarco_100k.json")

# 2. 定义你要的数据集样本
dataset_name = "ms_marco"
dataset_version = "v1.1"


dataset_split = "train[:100000]" 

# --- 脚本执行 ---

try:
    # 3. 确保 'data' 文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    print(f"确保 '{output_folder}' 文件夹存在...")

    # 4. 加载数据集样本
    print(f"正在加载数据集: {dataset_name} ({dataset_version}), 拆分: {dataset_split}...")
    print("这可能需要几分钟时间，具体取决于你的网络速度（下载量比之前大）。")
    
    dataset_sample = load_dataset(
        dataset_name, 
        dataset_version, 
        split=dataset_split,
        trust_remote_code=True
    )
    
    print("\n数据集样本加载成功！")
    print(f"样本包含 {len(dataset_sample)} 条记录。")

    # 5. 将数据集保存为 JSON 文件
    print(f"正在将数据保存为 JSON 格式到: {output_filename}...")
    
    dataset_sample.to_json(output_filename)
    
    print("\n--- 操作完成！ ---")
    print(f"文件已成功保存至: {output_filename}")

except Exception as e:
    print(f"\n--- 发生错误 ---")
    print(f"加载或保存数据时出错: {e}")