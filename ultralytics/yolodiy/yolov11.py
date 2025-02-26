import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义CSPBlock（Cross Stage Partial Block）
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.blocks = nn.Sequential(
            *[ResBlock(out_channels // 2) for _ in range(num_blocks)]
        )
        self.concat_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.concat_conv(torch.cat([y1, y2], dim=1))

# 定义ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))

# 定义SPPBlock（Spatial Pyramid Pooling Block）
class SPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1))

# 定义YOLOv11主干网络
class YOLOv11Backbone(nn.Module):
    def __init__(self):
        super(YOLOv11Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.csp1 = CSPBlock(32, 64, num_blocks=1)
        self.csp2 = CSPBlock(64, 128, num_blocks=3)
        self.spp = SPPBlock(128, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.spp(x)
        return x

# 定义YOLOv11检测头
class YOLOv11Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YOLOv11Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, (5 + num_classes) * 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# 定义YOLOv11完整模型
class YOLOv11(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv11, self).__init__()
        self.backbone = YOLOv11Backbone()
        self.head = YOLOv11Head(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

# 测试模型
if __name__ == "__main__":
    model = YOLOv11(num_classes=80)  # 假设80类目标
    x = torch.randn(1, 3, 416, 416)  # 输入图片大小为416x416
    outputs = model(x)
    print("输出维度：", outputs.shape)
