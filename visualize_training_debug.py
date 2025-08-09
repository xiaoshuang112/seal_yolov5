import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.yolo import YOLOv5s
from models.loss import ComputeLoss
from augmentation import YOLOv5Augmentation
from params_yolo import get_default_config
import seaborn as sns

def load_model(model_path, device='cpu'):
    """加载训练好的模型"""
    model = YOLOv5s().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
    
    return img_tensor, img, ratio, (dw, dh)

def analyze_confidence_distribution(pred_outputs, save_path="debug_output/confidence_distribution.png"):
    """分析置信度分布"""
    os.makedirs("debug_output", exist_ok=True)
    
    all_confs = []
    for pred in pred_outputs:
        # pred shape: [batch, anchors, height, width, channels]
        conf = torch.sigmoid(pred[..., 4])  # 应用sigmoid到置信度
        all_confs.extend(conf.flatten().cpu().numpy())
    
    plt.figure(figsize=(12, 4))
    
    # 置信度分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(all_confs, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.axvline(0.01, color='red', linestyle='--', label='thresh=0.01')
    plt.axvline(0.05, color='orange', linestyle='--', label='thresh=0.05')
    plt.legend()
    
    # 置信度累积分布
    plt.subplot(1, 3, 2)
    sorted_confs = np.sort(all_confs)
    cumulative = np.arange(1, len(sorted_confs) + 1) / len(sorted_confs)
    plt.plot(sorted_confs, cumulative)
    plt.xlabel('Confidence Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Confidence CDF')
    plt.axvline(0.01, color='red', linestyle='--', label='thresh=0.01')
    plt.axvline(0.05, color='orange', linestyle='--', label='thresh=0.05')
    plt.legend()
    
    # 置信度统计
    plt.subplot(1, 3, 3)
    stats = {
        'Mean': np.mean(all_confs),
        'Std': np.std(all_confs),
        'Max': np.max(all_confs),
        'Min': np.min(all_confs),
        '>0.01': np.sum(np.array(all_confs) > 0.01) / len(all_confs),
        '>0.05': np.sum(np.array(all_confs) > 0.05) / len(all_confs),
        '>0.1': np.sum(np.array(all_confs) > 0.1) / len(all_confs)
    }
    
    y_pos = np.arange(len(stats))
    plt.barh(y_pos, list(stats.values()))
    plt.yticks(y_pos, list(stats.keys()))
    plt.xlabel('Value')
    plt.title('Confidence Statistics')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"置信度分布图已保存到: {save_path}")
    print(f"置信度统计: mean={stats['Mean']:.4f}, std={stats['Std']:.4f}, >0.01: {stats['>0.01']:.2%}")
    
    return stats

def analyze_box_coordinates(pred_outputs, save_path="debug_output/box_coordinate_distribution.png"):
    """分析预测框坐标分布"""
    os.makedirs("debug_output", exist_ok=True)
    
    all_boxes = []
    all_wh = []
    
    for pred in pred_outputs:
        # pred shape: [batch, anchors, height, width, channels]
        boxes = pred[..., :4]  # x, y, w, h
        conf = torch.sigmoid(pred[..., 4])
        
        # 只分析置信度>0.01的框
        mask = conf > 0.01
        valid_boxes = boxes[mask]
        
        if len(valid_boxes) > 0:
            all_boxes.extend(valid_boxes.cpu().numpy())
            all_wh.extend(valid_boxes[:, 2:4].cpu().numpy())  # w, h
    
    if len(all_boxes) == 0:
        print("⚠️ 没有置信度>0.01的预测框！这是问题的核心！")
        return
    
    all_boxes = np.array(all_boxes)
    all_wh = np.array(all_wh)
    
    plt.figure(figsize=(15, 10))
    
    # x, y 坐标分布
    plt.subplot(2, 3, 1)
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.5, s=1)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('XY Coordinate Distribution')
    
    # w, h 尺寸分布
    plt.subplot(2, 3, 2)
    plt.scatter(all_wh[:, 0], all_wh[:, 1], alpha=0.5, s=1)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Width-Height Distribution')
    
    # x 坐标直方图
    plt.subplot(2, 3, 3)
    plt.hist(all_boxes[:, 0], bins=50, alpha=0.7, color='red')
    plt.xlabel('X coordinate')
    plt.ylabel('Count')
    plt.title('X Distribution')
    
    # y 坐标直方图
    plt.subplot(2, 3, 4)
    plt.hist(all_boxes[:, 1], bins=50, alpha=0.7, color='green')
    plt.xlabel('Y coordinate')
    plt.ylabel('Count')
    plt.title('Y Distribution')
    
    # w 尺寸直方图
    plt.subplot(2, 3, 5)
    plt.hist(all_wh[:, 0], bins=50, alpha=0.7, color='blue')
    plt.xlabel('Width')
    plt.ylabel('Count')
    plt.title('Width Distribution')
    
    # h 尺寸直方图
    plt.subplot(2, 3, 6)
    plt.hist(all_wh[:, 1], bins=50, alpha=0.7, color='purple')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('Height Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"坐标分布图已保存到: {save_path}")
    print(f"有效预测框数量: {len(all_boxes)}")
    print(f"坐标范围: x=[{all_boxes[:, 0].min():.2f}, {all_boxes[:, 0].max():.2f}], y=[{all_boxes[:, 1].min():.2f}, {all_boxes[:, 1].max():.2f}]")
    print(f"尺寸范围: w=[{all_wh[:, 0].min():.2f}, {all_wh[:, 0].max():.2f}], h=[{all_wh[:, 1].min():.2f}, {all_wh[:, 1].max():.2f}]")

def analyze_ground_truth_distribution(data_dir, save_path="debug_output/ground_truth_distribution.png"):
    """分析真实标签分布"""
    os.makedirs("debug_output", exist_ok=True)
    
    label_dir = os.path.join(data_dir, "labels")
    if not os.path.exists(label_dir):
        print(f"标签目录不存在: {label_dir}")
        return
    
    all_boxes = []
    all_wh = []
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        all_boxes.append([x, y, w, h])
                        all_wh.append([w, h])
        except Exception as e:
            print(f"读取标签文件失败 {label_file}: {e}")
    
    if len(all_boxes) == 0:
        print("没有找到有效的标签数据！")
        return
    
    all_boxes = np.array(all_boxes)
    all_wh = np.array(all_wh)
    
    plt.figure(figsize=(15, 10))
    
    # x, y 坐标分布
    plt.subplot(2, 3, 1)
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.5, s=10, color='red')
    plt.xlabel('X coordinate (normalized)')
    plt.ylabel('Y coordinate (normalized)')
    plt.title('GT XY Coordinate Distribution')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # w, h 尺寸分布
    plt.subplot(2, 3, 2)
    plt.scatter(all_wh[:, 0], all_wh[:, 1], alpha=0.5, s=10, color='blue')
    plt.xlabel('Width (normalized)')
    plt.ylabel('Height (normalized)')
    plt.title('GT Width-Height Distribution')
    
    # x 坐标直方图
    plt.subplot(2, 3, 3)
    plt.hist(all_boxes[:, 0], bins=20, alpha=0.7, color='red')
    plt.xlabel('X coordinate')
    plt.ylabel('Count')
    plt.title('GT X Distribution')
    
    # y 坐标直方图
    plt.subplot(2, 3, 4)
    plt.hist(all_boxes[:, 1], bins=20, alpha=0.7, color='green')
    plt.xlabel('Y coordinate')
    plt.ylabel('Count')
    plt.title('GT Y Distribution')
    
    # w 尺寸直方图
    plt.subplot(2, 3, 5)
    plt.hist(all_wh[:, 0], bins=20, alpha=0.7, color='blue')
    plt.xlabel('Width')
    plt.ylabel('Count')
    plt.title('GT Width Distribution')
    
    # h 尺寸直方图
    plt.subplot(2, 3, 6)
    plt.hist(all_wh[:, 1], bins=20, alpha=0.7, color='purple')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('GT Height Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"真实标签分布图已保存到: {save_path}")
    print(f"总标签数量: {len(all_boxes)}")
    print(f"标签坐标范围: x=[{all_boxes[:, 0].min():.3f}, {all_boxes[:, 0].max():.3f}], y=[{all_boxes[:, 1].min():.3f}, {all_boxes[:, 1].max():.3f}]")
    print(f"标签尺寸范围: w=[{all_wh[:, 0].min():.3f}, {all_wh[:, 0].max():.3f}], h=[{all_wh[:, 1].min():.3f}, {all_wh[:, 1].max():.3f}]")
    
    return all_wh

def comprehensive_training_analysis(model_path, img_path, data_dir, device='cpu'):
    """综合训练分析"""
    print("=" * 60)
    print("开始comprehensive训练诊断分析...")
    print("=" * 60)
    
    # 1. 加载模型和图片
    model = load_model(model_path, device)
    img_tensor, img_bgr, ratio, pad = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    
    # 2. 加载对应标签
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    targets = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                targets.append([0, cls, x, y, w, h])  # batch_idx=0
        targets = torch.tensor(targets, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"❌ 标签读取失败: {e}")
        targets = torch.zeros((0, 6), dtype=torch.float32).to(device)
    
    print(f"📁 分析图片: {img_path}")
    print(f"📋 标签数量: {len(targets)}")
    
    # 3. 模型推理
    model.train()
    with torch.no_grad():
        pred = model(img_tensor)
    
    # 4. 损失分析
    if isinstance(pred, list) and len(pred) >= 3 and isinstance(pred[2], list):
        pred_for_loss = pred[2]
    else:
        print("⚠️ 模型输出格式异常，使用原始输出")
        pred_for_loss = pred
    
    # 构建损失函数
    hyp = {
        'box': 0.05, 'obj': 1.0, 'cls': 0.5,
        'cls_pw': 1.0, 'obj_pw': 1.0,
        'anchor_t': 4.0, 'fl_gamma': 0.0,
        'label_smoothing': 0.0
    }
    compute_loss = ComputeLoss(model, hyp)
    
    try:
        loss_result = compute_loss(pred_for_loss, targets)
        if isinstance(loss_result, tuple) and len(loss_result) == 2:
            total_loss, loss_items = loss_result
            lbox, lobj, lcls = loss_items
            print(f"🔍 损失分析: total={total_loss.item():.4f}, lbox={lbox.item():.4f}, lobj={lobj.item():.4f}, lcls={lcls.item():.4f}")
            print(f"📊 损失权重: box={hyp['box']}, obj={hyp['obj']}, cls={hyp['cls']}")
            print(f"⚖️ 加权损失: lbox_weighted={lbox.item()*hyp['box']:.4f}, lobj_weighted={lobj.item()*hyp['obj']:.4f}, lcls_weighted={lcls.item()*hyp['cls']:.4f}")
        else:
            print(f"❌ 损失计算格式异常: {type(loss_result)}")
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
    
    # 5. 置信度分布分析
    print("\n🔍 分析置信度分布...")
    conf_stats = analyze_confidence_distribution(pred_for_loss)
    
    # 6. 预测框坐标分布分析
    print("\n🔍 分析预测框坐标分布...")
    analyze_box_coordinates(pred_for_loss)
    
    # 7. 真实标签分布分析
    print("\n🔍 分析真实标签分布...")
    gt_wh = analyze_ground_truth_distribution(data_dir)
    
    # 8. 深度问题诊断
    print("\n" + "=" * 60)
    print("🔬 深度问题诊断报告:")
    print("=" * 60)
    
    # 诊断1: 置信度塌陷
    if conf_stats['Mean'] < 0.01:
        print("❌ 问题1: 置信度严重塌陷！")
        print(f"   - 平均置信度仅 {conf_stats['Mean']:.6f}，远低于正常范围(0.1-0.5)")
        print(f"   - {conf_stats['>0.01']:.2%} 的预测框置信度 > 0.01")
        print("   🔧 建议: 降低obj损失权重，从1.0调整到0.3-0.5")
    elif conf_stats['Mean'] < 0.05:
        print("⚠️ 问题1: 置信度偏低")
        print(f"   - 平均置信度 {conf_stats['Mean']:.4f}，建议提升到0.1以上")
        print("   🔧 建议: 适当降低obj损失权重")
    else:
        print("✅ 置信度分布正常")
    
    # 诊断2: 损失权重不平衡
    if hyp['obj'] / hyp['box'] > 10:
        print("❌ 问题2: 损失权重严重不平衡！")
        print(f"   - obj权重({hyp['obj']})是box权重({hyp['box']})的{hyp['obj']/hyp['box']:.1f}倍")
        print("   - 模型过度关注置信度，忽略定位精度")
        print("   🔧 建议: 调整为 box=0.1, obj=0.5, cls=0.5")
    
    # 诊断3: 预测框坐标异常
    if len(targets) > 0:
        # 计算真实框的统计信息
        gt_boxes = targets[:, 2:6]  # x, y, w, h
        gt_wh_mean = gt_boxes[:, 2:4].mean(dim=0)
        print(f"📏 真实框平均尺寸: w={gt_wh_mean[0]:.3f}, h={gt_wh_mean[1]:.3f}")
        
        if gt_wh_mean[0] < 0.1 or gt_wh_mean[1] < 0.1:
            print("⚠️ 问题3: 目标尺寸很小，可能需要调整anchor")
        elif gt_wh_mean[0] > 0.8 or gt_wh_mean[1] > 0.8:
            print("⚠️ 问题3: 目标尺寸很大，可能需要调整anchor")
    
    print("\n🎯 核心问题总结:")
    print("1. 如果loss下降但IoU不升，最可能的原因是置信度塌陷")
    print("2. obj损失权重过高会导致模型'不敢'预测高置信度")
    print("3. 建议优先调整损失权重: obj从1.0降到0.3-0.5")
    print("4. 然后检查anchor是否与目标尺寸匹配")
    
    return conf_stats, gt_wh

def visualize_predictions_vs_gt(model_path, img_path, device='cpu', conf_thresh=0.01):
    """可视化预测框vs真实框对比"""
    model = load_model(model_path, device)
    img_tensor, img_bgr, ratio, pad = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    
    # 加载标签
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    gt_boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                gt_boxes.append([x, y, w, h])
    except:
        gt_boxes = []
    
    # 推理
    model.train()
    with torch.no_grad():
        pred = model(img_tensor)
    
    # 提取预测结果
    if isinstance(pred, list) and len(pred) >= 3 and isinstance(pred[2], list):
        pred_for_vis = pred[2]
        # 拼接多个检测头
        batch_size = pred_for_vis[0].shape[0]
        all_preds = []
        for p in pred_for_vis:
            p_reshaped = p.view(batch_size, -1, p.shape[-1])
            all_preds.append(p_reshaped)
        pred_cat = torch.cat(all_preds, 1)[0]  # [num_anchors, 6]
    else:
        pred_cat = pred[0]
    
    # 应用sigmoid到置信度
    pred_cat[:, 4] = torch.sigmoid(pred_cat[:, 4])
    
    # 过滤预测框
    conf_mask = pred_cat[:, 4] > conf_thresh
    valid_preds = pred_cat[conf_mask]
    
    # 创建可视化图像
    img_vis = img_bgr.copy()
    h, w = img_vis.shape[:2]
    
    # 画真实框 (绿色)
    for gt_box in gt_boxes:
        x, y, w_gt, h_gt = gt_box
        x1 = int((x - w_gt/2) * w)
        y1 = int((y - h_gt/2) * h)
        x2 = int((x + w_gt/2) * w)
        y2 = int((y + h_gt/2) * h)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_vis, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 画预测框 (红色)
    pred_count = 0
    for pred_box in valid_preds:
        x, y, w_pred, h_pred, conf, cls = pred_box
        # 这里坐标可能需要归一化处理，取决于模型输出格式
        x1 = int(max(0, min((x - w_pred/2) * w, w)))
        y1 = int(max(0, min((y - h_pred/2) * h, h)))
        x2 = int(max(0, min((x + w_pred/2) * w, w)))
        y2 = int(max(0, min((y + h_pred/2) * h, h)))
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_vis, f'{conf:.3f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        pred_count += 1
        
        if pred_count >= 20:  # 限制显示框数，避免太乱
            break
    
    # 保存结果
    output_path = "debug_output/predictions_vs_gt.jpg"
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite(output_path, img_vis)
    
    print(f"预测框 vs 真实框对比图已保存到: {output_path}")
    print(f"真实框数量: {len(gt_boxes)}, 有效预测框数量: {len(valid_preds)}")

if __name__ == "__main__":
    # 配置
    model_path = "model/best_model.pt"
    img_path = "data/images/20250714103802_00000.jpg"
    data_dir = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 开始训练诊断分析...")
    
    # 执行综合分析
    comprehensive_training_analysis(model_path, img_path, data_dir, device)
    
    # 执行可视化对比
    print("\n🖼️ 生成预测框vs真实框对比图...")
    visualize_predictions_vs_gt(model_path, img_path, device, conf_thresh=0.01)
    
    print("\n✅ 分析完成！请查看debug_output目录下的分析结果图片。")
