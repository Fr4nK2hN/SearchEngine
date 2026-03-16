import os

from datasets import load_dataset


def main():
    output_folder = "data"
    output_filename = os.path.join(output_folder, "msmarco_100k.json")
    dataset_name = "ms_marco"
    dataset_version = "v1.1"
    dataset_split = "train[:100000]"

    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"确保 '{output_folder}' 文件夹存在...")

        print(f"正在加载数据集: {dataset_name} ({dataset_version}), 拆分: {dataset_split}...")
        print("这可能需要几分钟时间，具体取决于你的网络速度（下载量比之前大）。")

        dataset_sample = load_dataset(
            dataset_name,
            dataset_version,
            split=dataset_split,
            trust_remote_code=True,
        )

        print("\n数据集样本加载成功！")
        print(f"样本包含 {len(dataset_sample)} 条记录。")

        print(f"正在将数据保存为 JSON 格式到: {output_filename}...")
        dataset_sample.to_json(output_filename)

        print("\n--- 操作完成！ ---")
        print(f"文件已成功保存至: {output_filename}")

    except Exception as exc:
        print("\n--- 发生错误 ---")
        print(f"加载或保存数据时出错: {exc}")
