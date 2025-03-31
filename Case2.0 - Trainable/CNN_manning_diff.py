# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:52:36 2025

@author: ASUS
"""

import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self, shapeX):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(shapeX, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 非 in-place 操作
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 非 in-place 操作
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 非 in-place 操作
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
        x = self.conv1(x)
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
