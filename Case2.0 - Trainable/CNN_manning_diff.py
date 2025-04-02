# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:52:36 2025

@author: ASUS
"""
import torch
import torch.nn as nn

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # 非 in-place 操作
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )
        # 如果输入输出通道不同，使用 1x1 卷积调整
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

# 修改后的 CNN1，加入了残差层
class CNN1(nn.Module):
    def __init__(self, shapeX):
        super(CNN1, self).__init__()
        # 初始卷积块：将输入通道转换为32，再转换为64
        self.initial = nn.Sequential(
            nn.Conv2d(shapeX, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 非 in-place 操作
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()   # 非 in-place 操作
        )
        # 残差层：保持通道数为64
        self.resblock = ResidualBlock(64, 64, kernel_size=1, stride=1, padding=0)
        # 后续卷积块
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 非 in-place 操作
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus()  # 确保输出始终为正，缓解数值不稳定问题
        )
        # fc 部分目前未在 forward 中使用，如有需要可展开使用
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),  # 非 in-place 操作
            nn.Linear(32, 16),
            nn.ReLU(),  # 非 in-place 操作
            nn.Linear(16, 8),
            nn.ReLU(),  # 非 in-place 操作
            nn.Linear(8, 4),
            nn.ReLU(),  # 非 in-place 操作
            nn.Linear(4, 3),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock(x)  # 添加残差连接层
        x = self.final(x)
        # 如果需要同时使用全连接层，可按如下方式：
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x   # 返回卷积层的输出

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

# import torch
# import torch.nn as nn

# class CNN1(nn.Module):
#     def __init__(self, shapeX):
#         super(CNN1, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(shapeX, 32, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#             nn.Softplus()  # 确保输出始终为正，缓解数值不稳定问题
#         )
#         # fc 部分目前未在 forward 中使用，如有需要可展开使用
#         self.fc = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Linear(32, 16),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Linear(16, 8),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Linear(8, 4),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Linear(4, 3),
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         # 如果需要同时使用全连接层，可按如下方式：
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         return x   # 返回卷积层的输出

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.ReLU(),  # 非 in-place 操作
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x
