# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLOv5s 轻量级模型，专为印章检测优化
"""

import torch
import torch.nn as nn
import math

# ===================== YOLOv5s 配置（直接内置，无需 yaml 文件） =====================
YOLOV5S_CFG = {
    'nc': 1,  # 类别数，按需修改
    'depth_multiple': 0.33,
    'width_multiple': 0.50,
    'anchors': [
        [38, 40, 57, 50, 67, 69],
        [84, 74, 95, 93, 129, 90],
        [112, 109, 143, 141, 238, 233],
    ],
    'backbone': [
        [-1, 1, 'Conv', [64, 6, 2, 2]],
        [-1, 1, 'Conv', [128, 3, 2]],
        [-1, 3, 'C3', [128]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [-1, 6, 'C3', [256]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [-1, 9, 'C3', [512]],
        [-1, 1, 'Conv', [1024, 3, 2]],
        [-1, 3, 'C3', [1024]],
        [-1, 1, 'SPPF', [1024, 5]],
    ],
    'head': [
        [-1, 1, 'Conv', [512, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],
        [-1, 3, 'C3', [512, False]],
        [-1, 1, 'Conv', [256, 1, 1]],
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],
        [-1, 3, 'C3', [256, False]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 14], 1, 'Concat', [1]],
        [-1, 3, 'C3', [512, False]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 10], 1, 'Concat', [1]],
        [-1, 3, 'C3', [1024, False]],
        [[17, 20, 23], 1, 'Detect', []],
    ],
}

# ===================== 基础模块 =====================
def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        c_ = int(out_channels * 0.5)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)  # 修正：cv2 输入通道应为 in_channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, 1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, out_channels, 1, 1)
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_k=5):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(pool_k, 1, pool_k // 2)
        self.conv2 = Conv(out_channels * 4, out_channels, 1)
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), 1))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        shapes = torch.stack([torch.tensor([t.shape[2], t.shape[3]], device=t.device) for t in x])
        min_hw = torch.amin(shapes, dim=0)
        out = []
        for t in x:
            h, w = t.shape[2], t.shape[3]
            if (h != min_hw[0]) or (w != min_hw[1]):
                t = torch.nn.functional.interpolate(t, size=(int(min_hw[0]), int(min_hw[1])), mode='nearest')
            out.append(t)
        return torch.cat(out, self.d)

class Detect(nn.Module):
    def __init__(self, nc=1, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors) if anchors else 3
        self.na = len(anchors[0]) // 2 if anchors else 3
        anchors = anchors if anchors else YOLOV5S_CFG['anchors']
        # 定义stride并将anchors标准化到网格单位（像素/stride），与训练/损失保持一致
        self.stride = torch.tensor([8., 16., 32.])
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) / self.stride.view(self.nl, 1, 1)
        self.m = nn.ModuleList([nn.Conv2d(x, self.no * self.na, 1) for x in ch])
        self.grid = [torch.zeros(1) for _ in range(self.nl)]
        self.anchor_grid = [torch.zeros(1) for _ in range(self.nl)]
    def forward(self, x):
        z = []
        bs = x[0].shape[0]
        for i in range(self.nl):
            out = self.m[i](x[i])
            h, w = out.shape[-2], out.shape[-1]
            if self.grid[i].numel() != h * w * 2:
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
                self.grid[i] = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).to(out.device)
                # 将标准化后的anchors乘以stride得到像素单位，用于解码
                self.anchor_grid[i] = (self.anchors[i].view(1, self.na, 1, 1, 2) * self.stride[i]).view(1, self.na, 1, 1, 2).to(out.device)
            out = out.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            if self.training:
                z.append(out)
            else:
                z.append(out.view(bs, -1, self.no))
        if self.training:
            return z
        return torch.cat(z, 1)

# ===================== 网络结构自动构建 =====================
class YOLOv5s(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = YOLOV5S_CFG if cfg is None else cfg
        self.model, self.save = self.parse_model(self.cfg)
        self.loss = None  # 可按需添加损失函数

    def parse_model(self, cfg):
        layers, save = [], []
        ch = [3]
        gd, gw = cfg['depth_multiple'], cfg['width_multiple']
        for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
            mtype = eval(m) if isinstance(m, str) else m
            n_ = max(round(n * gd), 1) if n > 1 else n
            if mtype is Concat:
                # Concat 层输入是 list，输出通道数是 sum([ch[x+1 if x!=-1 else -1] for x in f])，与forward索引一致
                in_ch = [ch[x+1 if x!=-1 else -1] for x in f]  # 仅用于后续调试
                layer = Concat(args[0] if args else 1)
                c2 = sum([ch[x+1 if x!=-1 else -1] for x in f])
            else:
                if isinstance(f, int):
                    in_ch = ch[f]
                elif isinstance(f, list):
                    in_ch = ch[f[0]]
                else:
                    in_ch = ch[-1]
                if mtype is Conv:
                    c2 = make_divisible(args[0] * gw, 8)
                    kernel_size = args[1] if len(args) > 1 else 3
                    stride = args[2] if len(args) > 2 else 1
                    layer = Conv(in_ch, c2, kernel_size, stride)
                elif mtype is C3:
                    c2 = make_divisible(args[0] * gw, 8)
                    layer = C3(in_ch, c2, n_, args[1] if len(args) > 1 else True)
                elif mtype is SPPF:
                    c2 = make_divisible(args[0] * gw, 8)
                    pool_k = args[1] if len(args) > 1 else 5
                    layer = SPPF(in_ch, c2, pool_k)
                elif mtype is nn.Upsample:
                    layer = nn.Upsample(scale_factor=args[1], mode=args[2])
                    c2 = in_ch
                elif mtype is Detect:
                    detect_ch = [ch[x+1 if x!=-1 else -1] for x in f]
                    layer = Detect(cfg['nc'], cfg['anchors'], detect_ch)
                    c2 = cfg['nc'] + 5
                else:
                    raise ValueError(f"Unknown module: {m}")
            layers.append(layer)
            ch.append(c2)
            if isinstance(f, int):
                save.append(f)
            elif isinstance(f, list):
                save.extend(f)
        return nn.ModuleList(layers), sorted(set(save))

    def forward(self, x, targets=None):
        outputs = [x]
        for i, layer in enumerate(self.model):
            f = YOLOV5S_CFG['backbone'] + YOLOV5S_CFG['head']
            f = f[i][0]
            if isinstance(f, int):
                inp = outputs[f + 1 if f != -1 else -1]
            elif isinstance(f, list):
                inp = [outputs[j + 1 if j != -1 else -1] for j in f]
            else:
                inp = outputs[-1]
            x = layer(inp)
            outputs.append(x)
        # 收集检测头的输出
        detection_outputs = []
        # 假设最后3层是检测头（根据YOLOv5架构）
        for i in range(-3, 0):
            if i < len(outputs) and outputs[i] is not None:
                detection_outputs.append(outputs[i])
        
        # 如果没有检测到合适的输出，使用最后一个输出
        if not detection_outputs:
            detection_outputs = [outputs[-1]] if outputs else []
            
        return detection_outputs

# ===================== 测试代码 =====================
if __name__ == "__main__":
    model = YOLOv5s()
    model.eval()
    dummy_input = torch.ones(1, 3, 640, 640)
    out = model(dummy_input)
    print("Output shape:", out.shape)

