import yaml
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

# 自动聚类anchor脚本，适用于YOLOv5标签格式
# 用法：python auto_anchor.py seal.yaml

def load_labels_from_yaml(yaml_path):
    """
    读取yaml文件，收集所有归一化标签 [cls, x, y, w, h]
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    labels = []
    for label_list in data.get('labels', []):
        for obj in label_list:
            # [cls, x, y, w, h]，归一化
            labels.append(obj)
    return labels

def auto_anchors(label_list, img_size=640, n_anchors=9):
    """
    label_list: 所有标签的列表，每个元素为 [cls, x, y, w, h]，归一化到[0,1]
    img_size: 输入图片尺寸
    n_anchors: anchor总数（YOLOv5默认9）
    """
    wh = []
    for label in label_list:
        w = label[3] * img_size
        h = label[4] * img_size
        wh.append([w, h])
    wh = np.array(wh)
    # k-means 聚类
    kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(wh)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 按面积排序
    print("建议 anchor (w, h):")
    for i, (w, h) in enumerate(anchors):
        print(f"[{w:.1f}, {h:.1f}]")
    # 分组输出，适配YOLOv5 yaml格式
    print("\nYOLOv5 anchors yaml格式：")
    print("anchors:")
    for i in range(3):
        group = anchors[i*3:(i+1)*3].reshape(-1)
        print(f"  - [{', '.join([f'{v:.0f}' for v in group])}]")
    return anchors

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # 直接调用 SealDataset，获取所有归一化标签
    from seal_dataset import SealDataset

    yaml_path = "data/seal.yaml"  # 如有需要可修改为你的实际路径
    dataset = SealDataset(yaml_path, img_size=640, augment=False, is_train=True)
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        # labels: tensor(N, 5) [cls, x, y, w, h]，已归一化
        for obj in labels:
            all_labels.append(obj.tolist())
    auto_anchors(all_labels, img_size=640, n_anchors=9)
