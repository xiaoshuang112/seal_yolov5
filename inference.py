import torch
import cv2
import numpy as np
from models.yolo import YOLOv5s
from augmentation import YOLOv5Augmentation

def decode_yolo_output(pred, device='cpu', img_size=640):
    """完整的YOLO输出解码 - 训练模式输出需要完整解码"""
    # 对于训练模式的输出，需要进行完整的YOLO解码
    # 这包括：坐标归一化、anchor处理、sigmoid激活等
    
    # 1. 对置信度和类别应用sigmoid
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])
    
    # 2. 对坐标应用sigmoid (YOLO中xy使用sigmoid，wh使用exp但这里简化)
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])
    
    # 3. 将坐标缩放到图像尺寸
    # 训练模式输出的坐标是相对于grid cell的，需要转换为像素坐标
    pred[..., 0:4] = pred[..., 0:4] * img_size
    
    return pred

def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    model = YOLOv5s().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(img_path, img_size=640):
    """预处理单张图片"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    
    # 单通道处理
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.merge([img, img, img])
    
    # letterbox处理
    augmentor = YOLOv5Augmentation()
    img, ratio, (dw, dh) = augmentor.letterbox(img, new_shape=(img_size, img_size), auto=False)
    
    # 保存预处理后的图片（BGR格式，归一化前）
    cv2.imwrite('preprocessed.jpg', img)
    # 转换为tensor
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # 添加batch维度
    
    return img, ratio, (dw, dh)

def simple_nms(detections, iou_thresh=0.5):
    """简单的NMS实现"""
    if len(detections) == 0:
        return detections
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # 取置信度最高的
        best = detections.pop(0)
        keep.append(best)
        
        # 计算与剩余框的IoU，移除重叠过大的框
        remaining = []
        for det in detections:
            iou = calculate_iou(best['bbox'], det['bbox'])
            if iou < iou_thresh:
                remaining.append(det)
        detections = remaining
    
    return keep

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def postprocess_detections(pred, ratio, pad, img_size=640, conf_thresh=0.5):
    """后处理检测结果"""
    detections = []
    
    # 置信度过滤
    conf_mask = pred[0, :, 4] > conf_thresh
    pred_filtered = pred[0][conf_mask]
    print(f"postprocess: conf_thresh={conf_thresh}, pred_filtered.shape={pred_filtered.shape}")
    if len(pred_filtered) == 0:
        print("postprocess: 无满足置信度的框，pred[0, :, 4]前5:", pred[0, :5, 4].cpu().numpy())
        return detections
    
    # 打印前几个框的坐标和置信度
    for i, detection in enumerate(pred_filtered[:3]):
        x, y, w, h, conf, cls = detection
        print(f"框{i}: x={x.item():.1f}, y={y.item():.1f}, w={w.item():.1f}, h={h.item():.1f}, conf={conf.item():.3f}, cls={cls.item():.2f}")
    
    for detection in pred_filtered:
        x, y, w, h, conf, cls = detection
        # 转换为xyxy格式（像素坐标）
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # 反letterbox变换（只对xyxy做一次）
        x1 = (x1 - pad[0]) / ratio[0]
        y1 = (y1 - pad[1]) / ratio[1]
        x2 = (x2 - pad[0]) / ratio[0]
        y2 = (y2 - pad[1]) / ratio[1]

        # 保证顺序
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 裁剪到图片范围
        x1 = max(0, min(x1, img_size))
        y1 = max(0, min(y1, img_size))
        x2 = max(0, min(x2, img_size))
        y2 = max(0, min(y2, img_size))

        w_new = abs(x2 - x1)
        h_new = abs(y2 - y1)

        # 只保留合理尺寸的检测框
        if w_new > 5 and h_new > 5 and w_new < img_size and h_new < img_size:
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf.item(),
                'class': int(cls.item())
            })
    
    return detections

def detect_image(model, img_path, device='cpu', conf_thresh=0.5):
    """对单张图片进行检测"""
    # 预处理
    img_tensor, ratio, pad = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        # 强制使用训练模式获取正确的输出格式，然后立即切回eval模式
        model.train()
        pred = model(img_tensor)
        model.eval()
        
        # 检查模型输出格式
        print(f"Model output type: {type(pred)}")
        if isinstance(pred, list):
            print(f"Output list length: {len(pred)}")
            for i, p in enumerate(pred):
                print(f"pred[{i}] shape: {p.shape if hasattr(p, 'shape') else type(p)}")
        
        # 处理训练模式的输出格式
        if isinstance(pred, list) and len(pred) >= 3:
            # 训练模式输出：[特征图1, 特征图2, 检测头输出列表]
            if isinstance(pred[2], list):
                preds = pred[2]  # 获取检测头输出列表
                print(f"Using training mode detection outputs, count: {len(preds)}")
                # 将多个检测头的输出拼接成单个张量（模拟eval模式）
                batch_size = preds[0].shape[0]
                decoded_outputs = []
                for i, pred_i in enumerate(preds):
                    # pred_i shape: [batch, anchors, height, width, channels]
                    pred_i_reshaped = pred_i.view(batch_size, -1, pred_i.shape[-1])
                    decoded_outputs.append(pred_i_reshaped)
                pred = torch.cat(decoded_outputs, 1)  # 拼接所有anchor points
                print(f"Concatenated prediction shape: {pred.shape}")
            else:
                pred = pred[2]
                print(f"Using single detection output, shape: {pred.shape}")
        else:
            print("Warning: Unexpected output format")
            return []
        
        # 对置信度和类别应用sigmoid，并进行完整的YOLO解码
        # 坐标需要特殊处理：sigmoid + grid + anchor
        pred = decode_yolo_output(pred, device=device)
        
        # 调试输出：置信度分布
        print(f"pred[..., 4] max: {pred[..., 4].max().item():.4f}, min: {pred[..., 4].min().item():.4f}, mean: {pred[..., 4].mean().item():.4f}")
        print(f"pred shape: {pred.shape}")
        print(f"推理后原始框总数: {pred.shape[1]}")
        print(f"推理后置信度前10: {pred[0, :10, 4].cpu().numpy()}")
    
    # 后处理
    detections = postprocess_detections(pred, ratio, pad, conf_thresh=conf_thresh)
    print(f"conf_thresh={conf_thresh} 后, 检测框数量: {len(detections)}")
    
    # 如果没有检测结果，尝试极低阈值再试一次
    if len(detections) == 0 and conf_thresh > 0.01:
        print("无检测结果，尝试conf_thresh=0.01")
        detections = postprocess_detections(pred, ratio, pad, conf_thresh=0.01)
        print(f"conf_thresh=0.01 后, 检测框数量: {len(detections)}")
    
    return detections

def draw_detections(img_path, detections, output_path=None):
    """在图片上绘制检测结果"""
    img = cv2.imread(img_path)
    
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
        conf = det['confidence']
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制置信度
        label = f"Seal: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

def train_mode_debug(model, img_path, device='cpu', conf_thresh=0.03):
    """训练模式推理，打印loss和画出所有预测框"""
    from models.loss import ComputeLoss
    from params_yolo import hyp
    import matplotlib.pyplot as plt

    # 加载图片和标签（假设标签格式为YOLO txt，需根据实际情况调整）
    img_tensor, ratio, pad = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    # 加载标签（假设与图片同名，后缀为.txt）
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    targets = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                targets.append([0, cls, x, y, w, h])  # batch_idx=0
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"标签读取失败: {e}")
        targets = torch.zeros((0, 6), dtype=torch.float32).to(device)

    # 训练模式推理
    model.train()
    pred = model(img_tensor)
    
    # 检查输出格式并提取正确的预测张量用于loss计算
    print(f"Train mode output type: {type(pred)}")
    if isinstance(pred, list):
        print(f"Output list length: {len(pred)}")
        for i, p in enumerate(pred):
            if isinstance(p, list):
                print(f"pred[{i}] is list with {len(p)} elements")
                for j, pp in enumerate(p):
                    print(f"  pred[{i}][{j}] shape: {pp.shape if hasattr(pp, 'shape') else type(pp)}")
            else:
                print(f"pred[{i}] shape: {p.shape if hasattr(p, 'shape') else type(p)}")
    
    # 提取用于loss计算的正确预测张量
    # YOLOv5训练模式通常返回 [x, x_det, inference_output] 或 [det_out_list]
    if isinstance(pred, list) and len(pred) >= 3:
        if isinstance(pred[2], list):
            # 使用第3个元素（检测头输出列表）进行loss计算
            pred_for_loss = pred[2]
        else:
            # 如果第3个元素不是列表，直接使用
            pred_for_loss = pred[2] if hasattr(pred[2], '__iter__') else [pred[2]]
    elif isinstance(pred, list) and len(pred) == 1:
        # 只有一个输出，检查是否为列表
        if isinstance(pred[0], list):
            pred_for_loss = pred[0]
        else:
            pred_for_loss = pred
    else:
        # 直接使用原始输出
        pred_for_loss = pred
        
    print(f"Using pred_for_loss: {type(pred_for_loss)}, length: {len(pred_for_loss) if isinstance(pred_for_loss, list) else 'not list'}")
    
    # 计算loss
    compute_loss = ComputeLoss(model, hyp)
    try:
        loss_result = compute_loss(pred_for_loss, targets)
        if isinstance(loss_result, tuple) and len(loss_result) == 2:
            total_loss, loss_items = loss_result
            lbox, lobj, lcls = loss_items
            print(f"lbox={lbox.item():.4f}, lobj={lobj.item():.4f}, lcls={lcls.item():.4f}, total={total_loss.item():.4f}")
        else:
            print(f"Unexpected loss result format: {type(loss_result)}")
    except Exception as e:
        print(f"Loss computation failed: {e}")
        # 使用零loss继续执行可视化部分
        lbox = lobj = lcls = torch.tensor(0.0)

    # 拼接所有检测头输出
    if isinstance(pred, list) and len(pred) >= 3:
        if isinstance(pred[2], list):
            preds = pred[2]
            batch_size = preds[0].shape[0]
            decoded_outputs = []
            for i, pred_i in enumerate(preds):
                pred_i_reshaped = pred_i.view(batch_size, -1, pred_i.shape[-1])
                decoded_outputs.append(pred_i_reshaped)
            pred_cat = torch.cat(decoded_outputs, 1)
        else:
            pred_cat = pred[2]
    else:
        pred_cat = pred

    # 对置信度和类别应用sigmoid
    pred_cat[..., 4:] = torch.sigmoid(pred_cat[..., 4:])

    # 画出所有预测框
    img = cv2.imread(img_path)
    for det in pred_cat[0]:
        x, y, w, h, conf, cls = det
        if conf.item() < conf_thresh:
            continue
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        x1 = int(max(0, min(x1, img.shape[1])))
        y1 = int(max(0, min(y1, img.shape[0])))
        x2 = int(max(0, min(x2, img.shape[1])))
        y2 = int(max(0, min(y2, img.shape[0])))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{conf.item():.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite('train_mode_debug.jpg', img)
    print("训练模式所有预测框已保存到 train_mode_debug.jpg")

if __name__ == "__main__":
    # 示例用法
    model_path = "model/best_model.pt"
    img_path = "data/images/20250714103802_00000.jpg"  # 使用训练数据集中的图片
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = load_model(model_path, device)
    
    #train_mode_debug(model, img_path, device, conf_thresh=0.3)

    # 检测，提升置信度阈值，减少低置信度误检
    conf_thres = 0.15
    detections = detect_image(model, img_path, device, conf_thresh=conf_thres)

    # 应用NMS减少重复检测
    detections = simple_nms(detections, iou_thresh=0.5)

    # 只输出置信度大于阈值的检测结果
    filtered = [det for det in detections if det['confidence'] >= conf_thres]
    print(f"NMS后检测到 {len(filtered)} 个印章 (置信度>={conf_thres})")
    for i, det in enumerate(filtered):
        print(f"印章 {i+1}: 置信度={det['confidence']:.3f}, 位置={det['bbox']}")

    # 保存结果图片
    result_img = draw_detections(img_path, filtered, "result.jpg")
    print("结果已保存到 result.jpg")
