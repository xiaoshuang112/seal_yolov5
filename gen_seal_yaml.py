import os
import yaml
import random


# 1. 获取图片和标注文件列表
def get_image_label_pairs(img_dir, label_dir):
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts]
    img_files.sort()
    pairs = []
    for img in img_files:
        base = os.path.splitext(img)[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if os.path.exists(label_path):
            pairs.append((os.path.join(img_dir, img).replace('\\', '/'), label_path.replace('\\', '/')))
    return pairs

# 2. 读取单个txt标注文件
def read_label_file(label_path, img_path):
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                x_min = float(parts[1])
                y_min = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                x_center = x_min + bw / 2
                y_center = y_min + bh / 2
                labels.append([cls, x_center, y_center, bw, bh])
    return labels

# 3. 生成yaml内容
def generate_yaml(pairs, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(pairs)
    idx = list(range(n))
    random.shuffle(idx)
    val_num = int(n * val_ratio)
    val_idx = set(idx[:val_num])
    val_pairs = [pairs[i] for i in range(n) if i in val_idx]
    # 训练集为全部图片
    train_pairs = pairs
    yaml_dict = {
        'train': [p[0] for p in train_pairs],
        'labels': [read_label_file(p[1], p[0]) for p in train_pairs],
        'val': [p[0] for p in val_pairs],
        'labels_val': [read_label_file(p[1], p[0]) for p in val_pairs]
    }
    return yaml_dict

if __name__ == '__main__':
    img_dir = 'data/images'
    label_dir = 'data/labels'
    out_yaml = 'data/seal.yaml'
    val_ratio = 0.8  # 测试集比例，可调整
    pairs = get_image_label_pairs(img_dir, label_dir)
    yaml_data = generate_yaml(pairs, val_ratio=val_ratio)
    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True)
    print(f'已生成: {out_yaml}, 总图片数: {len(pairs)}, 训练集: {len(yaml_data["train"])}, 测试集: {len(yaml_data["val"])}')
