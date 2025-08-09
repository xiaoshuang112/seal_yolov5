from dataclasses import dataclass
from typing import List, Tuple
from models.yolo import YOLOv5s as Model  # 假设 YOLOv5 模型定义文件

@dataclass
class TrainingConfig:
    """YOLOv5 印章检测模型的训练配置参数。"""

    # 设备配置
    gpu_num: str = "0"  # GPU设备编号，固定使用单张3090显卡

    # 数据配置
    img_size: int = 640  # 输入图像尺寸（宽度和高度）
    num_classes: int = 1  # 类别数（仅印章）
    data_yaml: str = "data/seal.yaml"  # 数据配置文件路径

    # 训练超参数
    lr: float = 0.01  # 初始学习率，建议使用0.01
    min_lr: float = 1e-6  # 最小学习率，建议设置为1e-6
    momentum: float = 0.9  # 动量参数
    weight_decay: float = 5e-4  # 权重衰减
    batch_size: int = 4  # 训练批次大小
    epochs: int = 50  # 训练轮次
    grad_clip_norm: float = 5.0  # 梯度裁剪阈值

    # 训练过程配置
    train_print_interval: int = 2  # 日志打印间隔（每多少步打印一次）
    save_interval: int = 5  # 模型保存间隔（每多少轮次保存一次）
    model_path: str = "./model"  # 模型保存目录

    # 模型配置
    model_name: str = "yolov5s"  # 模型名称
    model_dict = {
        "yolov5s": Model  # 使用 YOLOv5 的模型类
    }


def get_default_config() -> TrainingConfig:
    """返回默认配置对象。"""
    return TrainingConfig()

# 导出hyp字典，包含ComputeLoss所需的训练超参数
hyp = {
    'lr0': 0.01,  # 初始学习率
    'lrf': 0.01,  # 最小学习率（可选，部分实现可能需要）
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'warmup_epochs': 0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,  # box loss gain
    'cls': 0.5,   # cls loss gain
    'cls_pw': 1.0,
    'obj': 1.0,   # obj loss gain
    'obj_pw': 1.0,
    'iou_t': 0.20,  # IoU training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    'fl_gamma': 0.0,  # focal loss gamma
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
}

