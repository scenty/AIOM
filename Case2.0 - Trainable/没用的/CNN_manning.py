import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self, shapeX):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(shapeX, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # 输出通道为 1
        )

        # 输出单个数值的全连接层
        self.fc = nn.Sequential(
            nn.Linear(1, 1)  # 输入是 1，输出也是 1 个系数
        )

    def forward(self, x):
        x = self.conv1(x)               # shape: [B, 1, H, W]
        x = torch.mean(x, dim=[2, 3])   # 全局平均池化 => [B, 1]
        x = self.fc(x)                  # [B, 1] -> [B, 1]
        x = x.squeeze(-1)               # 去掉最后一维 => [B]
        return x  # 返回的是一个 batch-size 个系数


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),  # 非 in-place 操作
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
