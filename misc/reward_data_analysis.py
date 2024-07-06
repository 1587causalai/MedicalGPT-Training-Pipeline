import json
from collections import defaultdict


def analyze_dataset(file_path):
    print(f"正在分析文件: {file_path}")

    # 用于存储统计信息的变量
    total_samples = 0
    field_counts = defaultdict(int)
    field_types = defaultdict(set)
    field_lengths = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                total_samples += 1
                try:
                    sample = json.loads(line)
                    for key, value in sample.items():
                        field_counts[key] += 1
                        field_types[key].add(type(value).__name__)
                        if isinstance(value, str):
                            field_lengths[key].append(len(value))
                        elif isinstance(value, list):
                            field_lengths[key].append(len(value))
                except json.JSONDecodeError:
                    print(f"警告: 第 {total_samples} 行无法解析为 JSON")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return

    print(f"\n数据集分析结果:")
    print(f"总样本数: {total_samples}")

    print("\n字段统计:")
    for field, count in field_counts.items():
        print(f"  {field}:")
        print(f"    出现次数: {count}")
        print(f"    数据类型: {', '.join(field_types[field])}")
        if field_lengths[field]:
            avg_length = sum(field_lengths[field]) / len(field_lengths[field])
            print(f"    平均长度: {avg_length:.2f}")
            print(f"    最小长度: {min(field_lengths[field])}")
            print(f"    最大长度: {max(field_lengths[field])}")
        print()

    print("\n样本示例:")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sample = json.loads(next(file))
            print(json.dumps(sample, indent=2, ensure_ascii=False))
    except:
        print("无法读取样本示例")


if __name__ == "__main__":
    analyze_dataset("data/reward/dpo_zh_500.jsonl")