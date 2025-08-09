
# 避免 OpenMP 冲突报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
测试标签可视化功能
"""
from seal_dataset import SealDataset

print("测试标签可视化功能...")

# 创建数据集实例（开启可视化）
dataset = SealDataset('data/seal.yaml', augment=True)

# 启用可视化功能
dataset.visualize_labels = True

print("✅ 可视化功能已启用")
print("测试处理几个样本...")

# 处理前n个样本进行可视化
for i in range(len(dataset)):
    img, labels = dataset[i]
    print(f"处理样本 {i}: img.shape={img.shape}, labels.shape={labels.shape}")

print(f"\n✅ 可视化结果已保存到: debug_output/visualize_labels/")
print("📁 检查生成的图片文件:")

# 列出生成的文件
debug_dir = "debug_output/visualize_labels"
if os.path.exists(debug_dir):
    files = [f for f in os.listdir(debug_dir) if f.endswith('.jpg')]
    for file in sorted(files):
        print(f"  - {file}")
else:
    print("  (目录未创建)")

print("\n📝 使用说明:")
print("1. 训练时关闭可视化: dataset.visualize_labels = False")
print("2. 调试时开启可视化: dataset.visualize_labels = True")
print("3. 可视化文件保存在: debug_output/visualize_labels/")
print("4. 绿色框: 边界框")
print("5. 蓝色点: 中心点")
print("6. 绿色文字: 类别标签")
