import cv2
import numpy as np
import random
import math


class YOLOv5Augmentation:
    """基于YOLOv5原始实现的数据增强类"""
    
    def __init__(self, 
                 hsv_h=0.015, 
                 hsv_s=0.7, 
                 hsv_v=0.4, 
                 degrees=10.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 flipud=0.0,
                 fliplr=0.5,
                 mosaic=1.0):
        """
        初始化YOLOv5数据增强参数
        """
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
    
    def random_perspective(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=0, perspective=0.0,
                          border=(0, 0)):
        """
        YOLOv5原始random_perspective实现
        对图像和标签应用随机透视变换
        """
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets
    
    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
        """
        YOLOv5原始box_candidates实现
        计算候选框，基于宽高比、面积等条件过滤
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        """
        YOLOv5原始HSV增强实现
        """
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
        YOLOv5原始letterbox实现
        将图像resize到指定尺寸，保持宽高比
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def apply_augmentation(self, img, labels=None):
        """
        应用完整的YOLOv5数据增强流程
        
        Args:
            img: 输入图像 (numpy array)
            labels: 标签 [[cls, x, y, w, h], ...] 归一化坐标格式
            
        Returns:
            tuple: (增强后的图像, 增强后的标签)
        """
        height, width = img.shape[:2]
        
        # 将归一化标签转换为像素坐标格式 (xyxy)
        targets = np.array(labels) if labels is not None else np.array([])
        if len(targets) > 0:
            targets[:, 1] *= width   # x -> pixel
            targets[:, 2] *= height  # y -> pixel
            targets[:, 3] *= width   # w -> pixel
            targets[:, 4] *= height  # h -> pixel
            
            # 转换为xyxy格式用于random_perspective
            targets_xyxy = targets.copy()
            targets_xyxy[:, 1] = targets[:, 1] - targets[:, 3] / 2  # x1
            targets_xyxy[:, 2] = targets[:, 2] - targets[:, 4] / 2  # y1
            targets_xyxy[:, 3] = targets[:, 1] + targets[:, 3] / 2  # x2
            targets_xyxy[:, 4] = targets[:, 2] + targets[:, 4] / 2  # y2
        else:
            targets_xyxy = np.array([])

        # HSV增强
        if random.random() < 0.5:
            img = self.augment_hsv(img, hgain=self.hsv_h, sgain=self.hsv_s, vgain=self.hsv_v)

        # 随机透视变换 (包含旋转、缩放、剪切、平移等)
        if random.random() < 0.5:
            img, targets_xyxy = self.random_perspective(
                img, targets_xyxy,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective
            )

        # 随机水平翻转
        if random.random() < self.fliplr:
            img = np.fliplr(img)
            if len(targets_xyxy) > 0:
                targets_xyxy[:, [1, 3]] = width - targets_xyxy[:, [3, 1]]

        # 随机垂直翻转
        if random.random() < self.flipud:
            img = np.flipud(img)
            if len(targets_xyxy) > 0:
                targets_xyxy[:, [2, 4]] = height - targets_xyxy[:, [4, 2]]

        # 转换回归一化xywh格式
        result_labels = []
        if len(targets_xyxy) > 0:
            for target in targets_xyxy:
                cls = target[0]
                x1, y1, x2, y2 = target[1:5]
                x = (x1 + x2) / 2 / width
                y = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                # 边界检查
                if w > 0.01 and h > 0.01 and 0 <= x <= 1 and 0 <= y <= 1:
                    result_labels.append([cls, x, y, w, h])

        return img, result_labels
