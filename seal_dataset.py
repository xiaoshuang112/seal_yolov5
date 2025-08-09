import torch
from torch.utils.data import Dataset
import yaml
import cv2
import os
import numpy as np
from augmentation import YOLOv5Augmentation


class SealDataset(Dataset):
    """精简的海豹数据集类，专注于数据加载和基本处理"""
    
    def __init__(self, data_yaml, img_size=640, augment=False, is_train=True):
        """
        初始化数据集
        
        Args:
            data_yaml (str): 数据配置文件路径
            img_size (int): 输入图像尺寸
            augment (bool): 是否启用数据增强
            is_train (bool): 是否为训练集
        """
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        self.img_paths = data['train'] if is_train else data['val']
        self.labels = data['labels'] if is_train else data['labels_val']
        assert len(self.img_paths) == len(self.labels), "图片路径和标签数量不一致！"
        
        self.img_size = img_size
        self.augment = augment
        
        print(f"Dataset loaded: {len(self.img_paths)} images, augment={augment}")
        
        # 初始化数据增强器
        self.augmentor = YOLOv5Augmentation(
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.1,
            fliplr=0.5
        )

    def __len__(self):
        return len(self.img_paths)
    
    def _load_image(self, img_path):
        """加载和预处理图像"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件未找到: {img_path}")
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        
        # 处理不同的图像格式
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # RGBA -> RGB
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.merge([img, img, img])  # 灰度 -> RGB
            
        return img
    
    def _prepare_labels(self, labels, img_shape):
        """
        准备标签格式
        将像素坐标转换为归一化格式
        
        Args:
            labels: 原始标签 [cls, x_pixel, y_pixel, w_pixel, h_pixel]
            img_shape: 图像尺寸 (height, width)
        
        Returns:
            归一化后的标签 [cls, x_norm, y_norm, w_norm, h_norm]
        """
        height, width = img_shape
        normalized_labels = []
        
        for obj in labels:
            cls, x_pixel, y_pixel, w_pixel, h_pixel = obj
            
            # 转换为归一化坐标
            x_norm = float(x_pixel) / width
            y_norm = float(y_pixel) / height
            w_norm = float(w_pixel) / width
            h_norm = float(h_pixel) / height
            
            normalized_labels.append([float(cls), x_norm, y_norm, w_norm, h_norm])
        
        return normalized_labels
    
    def _apply_augmentation_pixel_coords(self, img, pixel_labels, img_shape):
        """
        直接在像素坐标上应用增强，避免重复转换
        
        Args:
            img: 输入图像
            pixel_labels: 像素坐标标签 [[cls, x_pixel, y_pixel, w_pixel, h_pixel], ...]
            img_shape: 图像尺寸 (height, width)
            
        Returns:
            tuple: (增强后图像, 增强后归一化标签)
        """
        height, width = img_shape
        
        # 将像素坐标转换为xyxy格式供YOLOv5增强使用
        targets_xyxy = []
        for obj in pixel_labels:
            cls, x_pixel, y_pixel, w_pixel, h_pixel = obj
            x1 = x_pixel - w_pixel / 2
            y1 = y_pixel - h_pixel / 2
            x2 = x_pixel + w_pixel / 2
            y2 = y_pixel + h_pixel / 2
            targets_xyxy.append([cls, x1, y1, x2, y2])
        
        targets_xyxy = np.array(targets_xyxy) if targets_xyxy else np.array([])
        
        # 应用YOLOv5增强 (直接在像素坐标上)
        if len(targets_xyxy) > 0:
            # HSV增强
            if np.random.random() < 0.5:
                img = self.augmentor.augment_hsv(img, hgain=self.augmentor.hsv_h, 
                                               sgain=self.augmentor.hsv_s, vgain=self.augmentor.hsv_v)

            # 随机透视变换 (包含旋转、缩放、剪切、平移等)
            if np.random.random() < 0.5:
                img, targets_xyxy = self.augmentor.random_perspective(
                    img, targets_xyxy,
                    degrees=self.augmentor.degrees,
                    translate=self.augmentor.translate,
                    scale=self.augmentor.scale,
                    shear=self.augmentor.shear,
                    perspective=self.augmentor.perspective
                )

            # 随机水平翻转
            if np.random.random() < self.augmentor.fliplr:
                img = np.fliplr(img)
                if len(targets_xyxy) > 0:
                    targets_xyxy[:, [1, 3]] = width - targets_xyxy[:, [3, 1]]

            # 随机垂直翻转
            if np.random.random() < self.augmentor.flipud:
                img = np.flipud(img)
                if len(targets_xyxy) > 0:
                    targets_xyxy[:, [2, 4]] = height - targets_xyxy[:, [4, 2]]
        
        # 转换回像素xywh格式（不归一化，主流程 letterbox 后统一归一化）
        result_labels = []
        if len(targets_xyxy) > 0:
            for target in targets_xyxy:
                cls = target[0]
                x1, y1, x2, y2 = target[1:5]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                # 边界检查（像素尺度，防止极小框）
                if w > 1 and h > 1:
                    result_labels.append([cls, x, y, w, h])
        return img, result_labels
    
    def _validate_labels(self, labels):
        """验证标签的有效性"""
        valid_labels = []
        for obj in labels:
            cls, x, y, w, h = obj
            # 检查标签是否在有效范围内
            if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                valid_labels.append(obj)
            else:
                # 尝试修复无效标签
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.01), 1.0)
                h = min(max(h, 0.01), 1.0)
                valid_labels.append([cls, x, y, w, h])
        
        return valid_labels if valid_labels else [[0, 0.5, 0.5, 0.1, 0.1]]  # 默认标签

    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.img_paths[idx]
        labels = self.labels[idx].copy()
        
        # 加载图像
        img = self._load_image(img_path)
        original_height, original_width = img.shape[:2]
        
        # 应用数据增强 - 直接在像素坐标上操作，避免重复转换
        # 1. 先做数据增强（增强时labels为像素坐标）
        if self.augment:
            img, labels = self._apply_augmentation_pixel_coords(img, labels, (original_height, original_width))
        # 2. letterbox前，labels为像素坐标（无增强时）
        # 3. letterbox处理
        img, ratio, (dw, dh) = self.augmentor.letterbox(
            img, new_shape=(self.img_size, self.img_size), auto=False, scaleFill=False
        )
        # 4. letterbox后，将标签像素坐标映射到新图像空间
        mapped_labels = []
        for obj in labels:
            cls, x, y, w, h = obj
            # 原始像素坐标（增强后或原始）映射到letterbox后像素坐标
            x = x * ratio[0] + dw
            y = y * ratio[1] + dh
            w = w * ratio[0]
            h = h * ratio[1]
            mapped_labels.append([cls, x, y, w, h])
        # 5. 归一化到letterbox后尺寸
        norm_labels = []
        for obj in mapped_labels:
            cls, x, y, w, h = obj
            norm_labels.append([cls, x / self.img_size, y / self.img_size, w / self.img_size, h / self.img_size])
        labels = norm_labels
        
        # 验证标签
        labels = self._validate_labels(labels)
        
        # 可视化调试开关 - 绘制标签框（训练时可关闭）
        if hasattr(self, 'visualize_labels') and self.visualize_labels:
            self._visualize_labels(img, labels, idx)
                
        # 转换为tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return img_tensor, labels
    
    def _visualize_labels(self, img, labels, idx):
        """
        可视化标签并保存图片
        Args:
            img: numpy.ndarray, 处理后的图片 (letterbox后)
            labels: 归一化标签 [[cls, x, y, w, h], ...]
            idx: 样本索引
        """
        img_vis = img.copy()
        h, w = img_vis.shape[:2]
        for obj in labels:
            cls, x_norm, y_norm, w_norm, h_norm = obj
            # 转换为真实坐标
            x_center = int(x_norm * w)
            y_center = int(y_norm * h)
            box_w = int(w_norm * w)
            box_h = int(h_norm * h)
            # 计算边界框坐标
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            # 绘制边界框
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制类别标签
            label_text = f"cls:{int(cls)}"
            cv2.putText(img_vis, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 绘制中心点
            cv2.circle(img_vis, (x_center, y_center), 3, (255, 0, 0), -1)
        # 保存可视化结果
        debug_dir = "debug_output/visualize_labels"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/sample_{idx:04d}.jpg", img_vis)
        




