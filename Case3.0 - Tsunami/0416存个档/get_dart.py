import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
from matplotlib.pyplot import *
import numpy as np
from scipy import interpolate

def download_and_process_dart_data(buoy_id,  start='2011-03-11 05:00', 
                      end='2011-03-11 17:00', days=.5):
    """
    下载并处理DART浮标数据
    
    参数:
        url: 数据文件URL
        start_time: 地震发生时间 (UTC)
        days: 要分析的天数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}t2011.txt.gz&dir=data/historical/dart/"
    # 创建统一时间轴 (5分钟间隔)
    common_time = pd.date_range(start, end, freq='5min').values
    common_sec = (common_time - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')
    
    # 跳过前2行注释
    print(buoy_id)
    df = pd.read_csv(url, delim_whitespace=True, skiprows=2, header=None,
                    names=['YY', 'MM', 'DD', 'hh', 'mm', 'ss', 'T', 'HEIGHT'])
    
    # 创建时间戳列
    df['datetime'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm', 'ss']].astype(str).agg(' '.join, axis=1),
                                    format='%Y %m %d %H %M %S')
    
    # 筛选时间范围
    end_time = pd.to_datetime(start) + pd.Timedelta(days=days)
    mask = (df['datetime'] >= start) & (df['datetime'] <= end)
    df = df.loc[mask].copy()
    
    # 计算海啸波高 (减去均值)
    time = df['datetime'].values
    h = df['HEIGHT'].values #- df['HEIGHT'].mean()
    h[h>9900]=np.nan
    h = h - np.nanmean(h)
    
    # 插值到统一时间轴
    t_sec = (time.astype('datetime64[s]').astype('float64'))
    interp_h = interpolate.interp1d(t_sec[~np.isnan(h)], h[~np.isnan(h)], 
                                  bounds_error=False, fill_value=np.nan)(common_sec)
    
    # # 绘制结果
    # plt.figure(figsize=(12, 5))
    # #plt.plot(time,h)
    # plt.plot(common_time, interp_h, label=f'Buoy {buoy_id}')
    # plt.axvline(pd.to_datetime(start), color='red', linestyle='--', label='Earthquake Time')
    # plt.xlabel('Time')
    # plt.ylabel('Wave Height (m)')
    # plt.title('DART Buoy Tsunami Wave Record')
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
        
    return common_time,interp_h


# 初始化浮标列表和经纬度矩阵
buoy_list = ['21418','21401','21413','21419','21414','21415','21416',
             '52402','46402','46403','46408','46409','46410','51407','52403','52406']

h    = []
for inx in range(len(buoy_list)):
    buoy_id = buoy_list[inx]
    time, tmp2 = download_and_process_dart_data(buoy_id)
        
    h.append(tmp2)

h_all = np.stack(h)
    
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
    ################high-pass！！！！
    h  = h - np.convolve(h, np.ones(12)/12, mode='same')
    h[:11]=np.nan
    h[-11:]=np.nan
    
    ax = plt.subplot(8, 2, i)
    ax.plot(time, h, label=f'Buoy {buoy_id}', linewidth=2)  # 进一步加粗曲线
    ax.axvline(pd.to_datetime('2011-03-11 05:46'), color='r', 
             linestyle='--', linewidth=1.5)  # 加粗参考线
    
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
plt.show()



buoy_info = pd.read_excel('浮标站位.xlsx')
lon = np.zeros((len(buoy_list)))  # 创建存储经纬度的矩阵
lat = np.zeros((len(buoy_list)))  # 创建存储经纬度的矩阵

# 匹配并获取每个浮标的经纬度
for i, buoy_id in enumerate(buoy_list):
    # 在excel中查找匹配的浮标
    match = buoy_info[buoy_info['浮标ID'] == int(buoy_id)]
    if not match.empty:
        lon[i] = match.iloc[0]['经度']
        lat[i] = match.iloc[0]['纬度']
        
mat_data = {
    'h_all': np.array(h_all),  # 确保是numpy数组
    'time': np.array(time),    # 确保是numpy数组
    'buoy_ids': buoy_list,
    'lon': lon, 'lat':lat
}

# 保存为.mat文件
import scipy.io as sio
sio.savemat('dart_data.mat', mat_data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
