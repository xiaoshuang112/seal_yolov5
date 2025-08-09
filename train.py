import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from models.loss import ComputeLoss, bbox_iou
import os
import time
import numpy as np
from params_yolo import TrainingConfig, get_default_config
from seal_dataset import SealDataset

def analyze_predictions_iou(preds, targets, conf_thresh=0.001):
    """分析预测框与真实框的IoU分布"""
    if targets.shape[0] == 0:
        return {"avg_iou": 0.0, "max_iou": 0.0, "matched_targets": 0, "total_preds": 0, "total_targets": 0}
    
    # 将预测结果转换为检测框格式
    all_pred_boxes = []
    all_pred_confs = []
    
    for i, pred in enumerate(preds):
        # pred shape: [batch, anchors, height, width, channels]
        batch_size = pred.shape[0]
        for b in range(batch_size):
            pred_batch = pred[b]  # [anchors, height, width, channels]
            
            # 重塑为 [num_anchors, channels]
            pred_reshaped = pred_batch.view(-1, pred_batch.shape[-1])
            
            # 应用sigmoid到置信度
            conf = torch.sigmoid(pred_reshaped[:, 4])
            
            # 过滤低置信度预测
            conf_mask = conf > conf_thresh
            if conf_mask.sum() == 0:
                continue
                
            valid_preds = pred_reshaped[conf_mask]
            valid_confs = conf[conf_mask]
            
            # 转换坐标 (假设已经是相对坐标)
            pred_boxes = valid_preds[:, :4].clone()
            # 将xywh转换为xyxy格式
            pred_boxes[:, 0] = valid_preds[:, 0] - valid_preds[:, 2] / 2  # x1
            pred_boxes[:, 1] = valid_preds[:, 1] - valid_preds[:, 3] / 2  # y1
            pred_boxes[:, 2] = valid_preds[:, 0] + valid_preds[:, 2] / 2  # x2
            pred_boxes[:, 3] = valid_preds[:, 1] + valid_preds[:, 3] / 2  # y2
            
            all_pred_boxes.append(pred_boxes)
            all_pred_confs.append(valid_confs)
    
    if len(all_pred_boxes) == 0:
        return {"avg_iou": 0.0, "max_iou": 0.0, "matched_targets": 0, "total_preds": 0, "total_targets": len(targets) if targets is not None else 0}
    
    all_pred_boxes = torch.cat(all_pred_boxes, 0)
    all_pred_confs = torch.cat(all_pred_confs, 0)
    
    # 获取真实框 (targets格式: [batch_idx, class, x, y, w, h])
    gt_boxes = []
    if targets.shape[0] > 0:
        for i in range(targets.shape[0]):
            target = targets[i]
            # 转换为xyxy格式
            gt_box = torch.zeros(4, device=targets.device)
            gt_box[0] = target[2] - target[4] / 2  # x1
            gt_box[1] = target[3] - target[5] / 2  # y1  
            gt_box[2] = target[2] + target[4] / 2  # x2
            gt_box[3] = target[3] + target[5] / 2  # y2
            gt_boxes.append(gt_box)
    
    if len(gt_boxes) == 0:
        return {"avg_iou": 0.0, "max_iou": 0.0, "matched_targets": 0, "total_preds": len(all_pred_boxes), "total_targets": 0}
    
    gt_boxes = torch.stack(gt_boxes, 0)
    
    # 计算所有预测框与所有真实框的IoU
    ious = []
    matched_targets = 0
    
    for gt_box in gt_boxes:
        if len(all_pred_boxes) > 0:
            # 计算当前真实框与所有预测框的IoU
            iou_matrix = bbox_iou(gt_box.unsqueeze(0), all_pred_boxes, xywh=False)
            max_iou = iou_matrix.max().item()
            ious.append(max_iou)
            if max_iou > 0.5:  # 匹配阈值
                matched_targets += 1
    
    avg_iou = np.mean(ious) if ious else 0.0
    max_iou = max(ious) if ious else 0.0
    
    return {
        "avg_iou": avg_iou,
        "max_iou": max_iou, 
        "matched_targets": matched_targets,
        "total_targets": len(gt_boxes),
        "total_preds": len(all_pred_boxes)
    }

def collate_fn(batch):
    """自定义collate函数，处理不同数量目标的标签"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = _process_batch_labels(labels)
    return imgs, targets

def _process_batch_labels(labels):
    """处理批次标签，添加batch索引"""
    targets = []
    for i, label in enumerate(labels):
        processed_label = _process_single_label(label, i)
        targets.append(processed_label)
    
    return torch.cat(targets, 0) if targets else torch.zeros((0, 6), dtype=torch.float32)

def _process_single_label(label, batch_idx):
    """处理单个标签，添加batch索引"""
    if len(label) > 0:
        # 确保label是2D张量
        if label.dim() == 1:
            label = label.unsqueeze(0)
        # 添加batch索引作为第一列
        batch_idx_tensor = torch.full((label.shape[0], 1), batch_idx, dtype=torch.float32)
        return torch.cat([batch_idx_tensor, label], dim=1)
    else:
        # 返回shape为(0,6)的空tensor，防止0-d tensor
        return torch.zeros((0, 6), dtype=torch.float32)

# 测试/验证函数
@torch.no_grad()
def test_model(model, dataloader, device, compute_loss):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_iou_stats = []
    
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        
        # 在验证时强制使用训练模式来获取正确的输出格式
        model.train()
        model_output = model(imgs)
        model.eval()  # 立即切回评估模式
        
        # 提取检测头输出（与训练时相同的逻辑）
        if isinstance(model_output, list) and len(model_output) >= 3:
            if isinstance(model_output[2], list):
                preds = model_output[2]
            else:
                preds = [model_output[2]]
        elif isinstance(model_output, list):
            preds = model_output
        else:
            preds = [model_output]
        
        # 计算损失
        loss, _ = compute_loss(preds, targets)
        total_loss += loss.item()
        num_batches += 1
        
        # 收集IoU统计信息
        iou_stats = analyze_predictions_iou(preds, targets, conf_thresh=0.01)
        all_iou_stats.append(iou_stats)
    
    avg_loss = total_loss / max(1, num_batches)
    
    # 计算平均IoU统计
    if all_iou_stats:
        avg_iou = np.mean([s['avg_iou'] for s in all_iou_stats])
        max_iou = np.max([s['max_iou'] for s in all_iou_stats])
        total_matched = sum([s['matched_targets'] for s in all_iou_stats])
        total_targets = sum([s['total_targets'] for s in all_iou_stats])
        total_preds = sum([s['total_preds'] for s in all_iou_stats])
        
        print(f"  Validation IoU: avg={avg_iou:.3f}, max={max_iou:.3f}, matched={total_matched}/{total_targets}, preds={total_preds}")
    
    return avg_loss

def train_model(model, dataloader, optimizer, device, config: TrainingConfig, val_loader=None, scheduler=None):
    """训练模型。"""
    model.train()
    best_val_loss = float('inf')
    # 构建损失函数（超参数hyp可自定义或从config获取）
    hyp = {
        'box': 0.2, 'obj': 0.1, 'cls': 0.5,  # 保持权重关系，先不大动
        'cls_pw': 1.0, 'obj_pw': 1.0,
        'anchor_t': 4.0, 'fl_gamma': 1.5,  # 启用Focal Loss缓解极端不平衡
        'label_smoothing': 0.0,
        'obj_iou_floor': 0.25  # 为正样本objectness设置IoU下限
     }
    compute_loss = ComputeLoss(model, hyp)

    for epoch in range(config.epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        try:
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                # 统计标签尺寸分布
                if targets.shape[0] > 0:
                    wh = targets[:,4:6]
                # 前向传播
                model_output = model(imgs)
                # 提取检测头输出
                if isinstance(model_output, list) and len(model_output) >= 3:
                    if isinstance(model_output[2], list):
                        preds = model_output[2]
                    else:
                        preds = [model_output[2]]
                elif isinstance(model_output, list):
                    preds = model_output
                else:
                    preds = [model_output]
                loss, loss_vec = compute_loss(preds, targets)
                lbox, lobj, lcls = loss_vec.cpu().numpy()
                ltotal = loss.item()
                # 分析IoU（每10个batch分析一次，避免影响训练速度）
                if batch_idx % (config.train_print_interval * 2) == 0:
                    with torch.no_grad():
                        iou_stats = analyze_predictions_iou(preds, targets, conf_thresh=0.01)
                        pred_conf_mean = torch.cat([torch.sigmoid(p[..., 4].reshape(-1)) for p in preds]).mean().item() if len(preds) > 0 else 0.0
                        # 计算高置信度预测比例
                        pred_confs = torch.cat([torch.sigmoid(p[..., 4].reshape(-1)) for p in preds])
                        high_conf_ratio = (pred_confs > 0.01).float().mean().item()
                        print(f"Epoch {epoch+1}/{config.epochs}, batch_idx {batch_idx}/{len(dataloader)}")
                        print(f"  Loss: {ltotal:.4f}, lbox={lbox:.4f}, lobj={lobj:.4f}, lcls={lcls:.4f}")
                        print(f"  IoU: avg={iou_stats['avg_iou']:.3f}, max={iou_stats['max_iou']:.3f}, matched={iou_stats['matched_targets']}/{iou_stats['total_targets']}, preds={iou_stats['total_preds']}")
                        print(f"  Pred conf: mean={pred_conf_mean:.4f}, >0.01={high_conf_ratio:.1%}")
                elif batch_idx % config.train_print_interval == 0:
                    print(f"Epoch {epoch+1}/{config.epochs}, batch_idx {batch_idx}/{len(dataloader)}, Loss: {ltotal:.4f}, lbox={lbox:.4f}, lobj={lobj:.4f}, lcls={lcls:.4f}")
                total_loss += ltotal
                num_batches += 1
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
        except Exception as e:
            print(f"训练中断: {e}")
            import traceback
            traceback.print_exc()
            break
        avg_train_loss = total_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        if val_loader is not None:
            model.eval()
            val_loss = test_model(model, val_loader, device, compute_loss)
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config.model_path, 'best_model.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved: {best_model_path}")
        if (epoch + 1) % config.save_interval == 0:
            save_path = os.path.join(config.model_path, f'yolov5_seal_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
        # 每个epoch结束时推进学习率调度器
        if scheduler is not None:
            scheduler.step()

def main():
    """主函数，执行训练流程。"""
    
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name(0))
    # 加载配置
    config = get_default_config()
    
    # 检查设备
    device = torch.device(f'cuda:{config.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = config.model_dict[config.model_name]().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据集（训练集关闭增强）
    dataset = SealDataset(config.data_yaml, img_size=config.img_size, augment=False, is_train=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                          num_workers=0, pin_memory=(device.type!='cpu'), drop_last=True, collate_fn=collate_fn)
    print(f"Training set loaded: {len(dataset)} images (augment=False)")
    
    # 加载预训练模型（如果有）
    pretrained_path = os.path.join(config.model_path, 'best_model.pt')
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Model loaded from {pretrained_path}")

    # 加载验证集
    val_dataset = SealDataset(config.data_yaml, img_size=config.img_size, augment=False, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                                  num_workers=0, pin_memory=(device.type!='cpu'), collate_fn=collate_fn)
    print(f"Validation set loaded: {len(val_dataset)} images")

    # 设置优化器和学习率调度器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=config.momentum, 
                         weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)

    # 创建模型保存目录
    os.makedirs(config.model_path, exist_ok=True)

    # 开始训练
    print(f"Starting training: {config.epochs} epochs, batch_size={config.batch_size}")
    train_model(model, dataloader, optimizer, device, config, val_loader, scheduler)

    # 保存最终模型
    final_save_path = os.path.join(config.model_path, 'yolov5_seal_final.pt')
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved: {final_save_path}")

if __name__ == '__main__':
    main()