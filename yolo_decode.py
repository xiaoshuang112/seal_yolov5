import numpy as np
import torch

def yolo_decode(pred, conf_thres=0.25, iou_thres=0.45, img_size=640, max_det=100):
    """
    pred: [N, 6] or [B, N, 6] (x, y, w, h, conf, cls)
    返回: [num_det, 6] (x1, y1, x2, y2, conf, cls)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if pred.ndim == 3:
        pred = pred[0]  # 只取batch=0
    # 置信度过滤
    mask = pred[:, 4] > conf_thres
    dets = pred[mask]
    if dets.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    # xywh -> xyxy
    boxes = np.zeros_like(dets[:, :4])
    boxes[:, 0] = dets[:, 0] - dets[:, 2] / 2  # x1
    boxes[:, 1] = dets[:, 1] - dets[:, 3] / 2  # y1
    boxes[:, 2] = dets[:, 0] + dets[:, 2] / 2  # x2
    boxes[:, 3] = dets[:, 1] + dets[:, 3] / 2  # y2
    scores = dets[:, 4]
    classes = dets[:, 5]
    # NMS
    keep = nms_numpy(boxes, scores, iou_thres, max_det)
    return np.concatenate([boxes[keep], scores[keep, None], classes[keep, None]], axis=1)

def nms_numpy(boxes, scores, iou_thres=0.45, max_det=100):
    """纯 numpy NMS，返回保留索引"""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)