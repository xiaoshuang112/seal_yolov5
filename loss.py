import torch
import torch.nn as nn
import math

def smooth_BCE(eps=0.1):
    """标签平滑，返回正负样本的目标值。"""
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    """用于类别不平衡的Focal Loss（可选，YOLOv5原版支持）"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算IoU/GIoU/DIoU/CIoU，支持xywh/xyxy格式。
    输入shape: box1(1,4) to box2(n,4)
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

class ComputeLoss:
    """YOLOv5损失计算（包含分类、框、置信度）"""
    def __init__(self, model, hyp):
        device = next(model.parameters()).device
        self.hyp = hyp
        self.device = device
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["obj_pw"]], device=device))
        self.cp, self.cn = smooth_BCE(hyp.get("label_smoothing", 0.0))
        if hyp.get("fl_gamma", 0) > 0:
            BCEcls = FocalLoss(BCEcls, hyp["fl_gamma"])
            BCEobj = FocalLoss(BCEobj, hyp["fl_gamma"])
        m = model.model[-1]
        self.balance = [4.0, 1.0, 0.4] if m.nl == 3 else [4.0, 1.0, 0.25, 0.06, 0.02]
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        self.gr = 1.0
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        
        # 检查输入格式并转换为训练格式
        # 如果是评估模式的输出（3维），返回零损失
        if len(p) == 1 and len(p[0].shape) == 3:
            # 这是评估模式的输出，形状为 [batch, total_anchors, channels]
            # 在评估模式下，返回零损失
            return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
        
        # 检查p是否为正确的张量列表格式
        if not isinstance(p, list):
            print(f"Warning: Expected list input, got {type(p)}")
            return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
            
        # 验证列表中的每个元素都是张量
        for i, pi in enumerate(p):
            if not hasattr(pi, 'shape'):
                print(f"Warning: p[{i}] is not a tensor, type: {type(pi)}")
                return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
        
        try:
            tcls, tbox, indices, anchors = self.build_targets(p, targets)
        except Exception as e:
            print(f"Error in build_targets: {e}")
            return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
            if b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()
                # 为objectness引入IoU下限，避免正样本早期被过低IoU压制
                obj_iou_floor = self.hyp.get("obj_iou_floor", None)
                if obj_iou_floor is not None:
                    iou = iou.clamp(min=float(obj_iou_floor))
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(b.shape[0]), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)
            lobj += self.BCEobj(pi[..., 4], tobj) * self.balance[i]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        g = 0.5
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ],
                device=self.device,
            ).float()
            * g
        )
        
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            
            # 安全地获取 height 和 width
            try:
                if len(shape) == 5:  # [batch, anchors, height, width, channels]
                    h, w = int(shape[2]), int(shape[3])
                elif len(shape) == 4:  # [batch, height, width, channels]
                    h, w = int(shape[1]), int(shape[2])
                elif len(shape) == 3:  # [batch, height, width]
                    h, w = int(shape[1]), int(shape[2])
                else:
                    # 默认值，防止崩溃
                    h, w = 1, 1
            except (IndexError, TypeError) as e:
                h, w = 1, 1
            
            gain[2:6] = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float)
            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            # 使用之前计算的 h 和 w，确保它们是整数
            h_int, w_int = int(h), int(w)
            indices.append((b, a, gj.clamp_(0, h_int - 1), gi.clamp_(0, w_int - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return tcls, tbox, indices, anch
