# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:39:52 2025

@author: Scenty
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
import pandas as pd

def plot_dart(time, h_all, buoy_list, t=None, model=None):
    plt.figure(figsize=(15, 20))  # 调整画布大小
    
    # 全局样式设置
    plt.rcParams.update({
        'font.size': 12,          # 增大基础字体
        'axes.titlesize': 14,      # 子图标题更大
        'axes.labelsize': 12,      # 坐标轴标签
        'xtick.labelsize': 10,     # x轴刻度
        'ytick.labelsize': 10,     # y轴刻度
        'lines.linewidth': 1.5     # 线宽增加
    })
    
    for i, h in enumerate(h_all, 1):

        buoy_id = buoy_list[i-1]
        ################high-pass
        # h  = h - np.convolve(h, np.ones(12)/12, mode='same')
        # h[:11]=np.nan
        # h[-11:]=np.nan
        
        ax = plt.subplot(8, 2, i)
        ax.plot(time, h, label=f'Buoy {buoy_id}', linewidth=2)  # 进一步加粗曲线
        ax.axvline(pd.to_datetime('2011-03-11 05:46'), color='r', 
                  linestyle='--', linewidth=1.5)  # 加粗参考线
        # 如果传入t和model，画模型结果
        if t is not None:
            ax.plot(time[:t],model[:,i-1],'r',linewidth=1)
        
        # 紧凑布局调整
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylabel('Wave (m)', labelpad=5)
        # 只在最下方和最左侧子图显示坐标轴标签
        if i < 15:  # 不是最下方子图
            ax.set_xticklabels([])        
            
        if i >= 15:  # 最下方子图
            ax.set_xlabel('Time', labelpad=5)
    
    plt.tight_layout(h_pad=1.5, w_pad=1.5)  # 行列间距控制
    plt.suptitle("Dart at an instant in time: " + f"{t*60}" + " second")
    plt.show()
    
    
if __name__=="__main__":
    
    from scipy.io import loadmat
    mat_data = loadmat(r'E:\Github\AIOM\Case3.5 - Tsunami Trainable\Data\dart_data_filtered.mat')
    time = mat_data['time'].squeeze(0)
    time = np.array([np.datetime64(x[0]) for x in time]) 
    #NumPy 对象数组（dtype=object），其中每个元素又是一个包含单个时间字符串的 NumPy 数组（dtype='<U19'）
    #需要先提取时间字符串，再统一转换为 datetime 类型。
      
    buoy_list=mat_data['buoy_ids']
    h_all = mat_data['h_all_highpass']
    
    plot_dart(time,h_all,buoy_list)
    