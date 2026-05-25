# -*- coding: utf-8 -*-
"""
DAS 背景 RMS 统计分析 - 用于计算阈值

基于原代码的通道分类和加权组合方法，计算参考背景数据的RMS统计特征
目的：为累积RMS分析提供合理的阈值参考

新增功能：支持手动指定通道组

@author: 13099
"""

import os
import numpy as np
import h5py
from scipy.signal import butter, filtfilt
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("RMS-Threshold")

# ================= 配置 =================
ref_file = r""
output_dir = r""

dataset_name = "/denoised"
fs = 1000.0
lowcut = 50.0
highcut = 180.0
order = 2

# 背景数据分析时间段（秒）
start_time = 130.0
end_time = 300.0

# RMS计算窗口
window_size_sec = 0.1

# ================= 通道分组模式配置 =================
# 分组模式: "auto" = 自动分组, "manual" = 手动指定通道组
grouping_mode = "manual"  # 可选: "auto" 或 "manual"

# ---------- 自动分组参数 ----------
# 仅在 grouping_mode = "auto" 时使用
channel_ranges = [
    [78, 176],
    [180, 258],
    [262, 320]
]
channel_step = 5
num_channels_per_group = 3

# ---------- 手动分组参数 ----------
# 仅在 grouping_mode = "manual" 时使用
# 每个元素是一个通道组，包含要加权组合的通道列表
# 示例格式：[[ch1, ch2, ch3], [ch4, ch5], ...]
manual_channel_groups = [
    # 示例：手动指定的通道组
    # [323],  
    # [324], 
    # [325], 
    # [326], 
    # [327],          # 组1
    # [328, 329, 330,331,332,333,334,335,336,337,338],        # 组2
    # [340], 
    # [341], 
    # [342], 
    # [343],        # 组3
    # [345, 346, 347,348,349,350,351,352,353,354,355],        # 组4
    # [357],  
    # [358],  
    # [359],  
    # [360],        # 组5
    # [362, 363, 364,365,367,368,369,370,371,372],        # 组6
    # [374], 
    # [375],
    # [376],
    # [377],       # 组7
    # [379, 380,381,382,383,384,385,386,387,388,389], 
    # [391],  
    # [392],   
    # [393],       
    # [394], 
    # [396,397,398,399,340,401,402,403,404,405,406],   # 组8
    # [408], 
    # [409], 
    # [410], 
    # [412,413,414,415,416,417,418,419,420,421,422], 
    # [424], 
    # [425], 
    # [426], 
    # [427], 
    # [428,429,430,431,432,433,434,435,436,437,438], 
    # [439], 
    # [440], 
    # [441], 
    # [442], 
    # [443,444,445,446,447,448,449,450,451,452,453], 
    # [455], 
    # [456], 
    # [457], 
    # [458], 
    # [459,460,461,462,463,464,465,466,467,468,469], 
    # [471], 
    # [472], 
    # [473], 
    # [474], 
    # [476,477,478,479,480,481,482,483,484,485,486], 
    # [488], 
    # [489], 
    # [490], 
    # [491],
    # [326],
    # [327],
    # [328],
    # [329],
    # [330],
    # [331,332,333,334,335,336,337,338,339,340,341],
    # [343],
    # [344],
    # [345],
    # [346],
    # [350,351,352,353,354,355,356,357,358,359,360],
    # [362],
    # [363],
    # [364],
    # [365],
    # [366,367,368,369,370,371,372,373,374,375,376],
    # [378],
    # [379],
    # [380],
    # [381],
    # [382,383,384,385,386,387,388,389,390,391,392],
    # [394],
    # [395],
    # [396],
    # [397],
    # [398,399,400,401,402,403,404,405,406,407,408],
    # [410],
    # [411],
    # [412],
    # [413,414,415,416,417,418,419,420,421,422,423],
    # [425],
    # [426],
    # [427],
    # [428],
    # [430,431,432,433,434,435,436,437,438,439,440],
    # [442],
    # [443],
    # [444],
    # [445],
    # [446,447,448,449,450,451,452,453,454,455,456],
    # [458],
    # [459],
    # [460],
    # [461],
    # [462,463,464,465,466,467,468,469,470,471,472],
    # [474],
    # [475],
    # [476],
    # [477],
    # [478,479,480,481,482,483,484,485,486,487,488],
    # [490],
    # [491],
    # [492],
    # [493],
    
    ###S2
    [332],  
    [333], 
    [334], 
    [335], 
    [336],          # 组1
    [337, 338, 339,340,341,342,343,344,345,346,347],
    
    [349],  
    [350], 
    [351], 
    [352],          # 组1
    [353, 354, 355,356,357,358,359,360,361,362,363],
    
    [365],  
    [366], 
    [367], 
    [368],          # 组1
    [369, 370, 371,372,373,374,375,376,377,378,379],
    
    [381],  
    [382], 
    [383], 
    [384],          # 组1
    [385, 386, 387,388,389,390,391,392,393,394,395],
    
    [397],  
    [398], 
    [399], 
    [400],          # 组1
    [401, 402, 403,404,405,406,407,408,409,410,411],
    
    [413],  
    [414], 
    [415], 
    [416],           # 组1
    [417, 418, 419,420,421,422,423,424,425,426,427],
    
    [429],  
    [430], 
    [431], 
    [432],          # 组1
    [433, 434, 435,436,437,438,439,440,441,442,443],
    
    [445],  
    [446], 
    [447], 
    [448],         # 组1
    [449, 450, 451,452,453,454,455,456,457,458,459],
    
    [461],  
    [462], 
    [463], 
    [464],           # 组1
    [465, 466, 467,468,469,470,471,472,473,474,475],
     
    [477], 
    [478], 
    [479],
    [480],         # 组1
    [481, 482,483,484,485,486,487,488,489,490,491],
    
    [493],  
    [494], 
    [495], 
    [496], 
    [497],          # 组1
]

# 手动分组时，可以选择是否按通道范围分类输出结果
# True: 按channel_ranges分类统计; False: 所有组作为一个整体统计
manual_group_by_range = False


# ================= 基础函数（复用原代码）=================
def bandpass_filter(data, fs, lowcut, highcut, order=2):
    """带通滤波"""
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = min(data.shape[1] - 1, 3 * (max(len(a), len(b)) - 1))
    return filtfilt(b, a, data, padlen=padlen)

def calc_noise_variance(ref_block, fs, start_time=None, end_time=None):
    """计算噪声方差"""
    total_points = ref_block.shape[0]
    start_idx = 0 if start_time is None else int(start_time * fs)
    end_idx = total_points if end_time is None else int(end_time * fs)
    ref_segment = ref_block[start_idx:end_idx, :]
    mean_vals = np.mean(ref_segment, axis=0, keepdims=True)
    var = np.mean((ref_segment - mean_vals)**2, axis=0)
    return var

def compute_weights(noise_var):
    """计算加权系数"""
    w = 1.0 / (noise_var + 1e-12)
    w /= np.sum(w)
    return w

def weighted_combine(block, weights):
    """加权组合信号"""
    return np.sum(block * weights[None, :], axis=1)

def compute_rms(signal, window_size):
    """计算滑动窗口RMS"""
    num_windows = max(1, (len(signal) - window_size) // window_size + 1)
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        if end > len(signal):
            break
        rms[i] = np.sqrt(np.mean(signal[start:end]**2))
    return rms

def load_hdf5_data(file_path, dataset_name, channel_range):
    """加载HDF5数据"""
    try:
        start_channel, end_channel = channel_range
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:, start_channel:end_channel+1]
        return data.astype(np.float32)
    except Exception as e:
        log.error(f"加载失败：{e}")
        return None

def load_hdf5_data_by_channels(file_path, dataset_name, channels):
    """根据指定通道列表加载HDF5数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            data = f[dataset_name][:, channels]
        return data.astype(np.float32)
    except Exception as e:
        log.error(f"加载失败：{e}")
        return None

def generate_channel_groups(channel_range, channel_step, num_channels_per_group):
    """生成通道组（滑动窗口）"""
    start_channel, end_channel = channel_range
    channel_groups = []
    for start_ch in range(start_channel, end_channel - num_channels_per_group + 2, channel_step):
        if start_ch + num_channels_per_group - 1 > end_channel:
            break
        channel_groups.append(list(range(start_ch, start_ch + num_channels_per_group)))
    return channel_groups


def get_channel_range_for_group(channels, channel_ranges):
    """
    根据通道组的第一个通道判断属于哪个通道范围
    返回范围索引（1-based）和范围本身，如果不属于任何范围返回 (0, None)
    """
    first_ch = channels[0]
    for idx, (start, end) in enumerate(channel_ranges, 1):
        if start <= first_ch <= end:
            return idx, [start, end]
    return 0, None


def validate_manual_groups(manual_groups, file_path, dataset_name):
    """验证手动通道组的有效性"""
    try:
        with h5py.File(file_path, 'r') as f:
            total_channels = f[dataset_name].shape[1]
        
        valid_groups = []
        for i, group in enumerate(manual_groups):
            if not group:
                log.warning(f"通道组 {i+1} 为空，已跳过")
                continue
            
            invalid_chs = [ch for ch in group if ch < 0 or ch >= total_channels]
            if invalid_chs:
                log.warning(f"通道组 {i+1} 包含无效通道 {invalid_chs}（有效范围: 0-{total_channels-1}），已跳过")
                continue
            
            valid_groups.append(group)
        
        return valid_groups
    except Exception as e:
        log.error(f"验证通道组失败：{e}")
        return []


# ================= 手动分组RMS统计分析 =================
def calculate_manual_group_rms_statistics(ref_file, manual_groups, channel_ranges,
                                          group_by_range, fs, lowcut, highcut,
                                          order, start_time, end_time, window_size_sec,
                                          dataset_name, output_dir):
    """计算手动指定通道组的RMS统计特征"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证通道组
    valid_groups = validate_manual_groups(manual_groups, ref_file, dataset_name)
    if not valid_groups:
        log.error("没有有效的通道组，退出")
        return []
    
    log.info(f"共 {len(valid_groups)} 个有效通道组")
    
    # 获取所有需要的通道
    all_channels = sorted(set(ch for group in valid_groups for ch in group))
    min_ch, max_ch = min(all_channels), max(all_channels)
    
    log.info(f"加载通道范围: {min_ch} - {max_ch}")
    
    # 加载数据
    ref_data = load_hdf5_data(ref_file, dataset_name, [min_ch, max_ch])
    if ref_data is None:
        return []
    
    # 滤波
    log.info(f"应用带通滤波 [{lowcut}-{highcut}] Hz...")
    ref_data = bandpass_filter(ref_data, fs, lowcut, highcut, order)
    
    # 提取分析时间段
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    ref_segment = ref_data[start_idx:end_idx, :]
    log.info(f"分析时间段: {start_time}-{end_time} 秒 ({ref_segment.shape[0]} 个样本点)")
    
    window_size = int(window_size_sec * fs)
    
    # 处理每个通道组
    all_results = []
    
    for g_idx, channels in enumerate(valid_groups):
        log.info(f"处理通道组 {g_idx+1}/{len(valid_groups)}: {channels}")
        
        # 获取通道索引（相对于加载的数据）
        channel_indices = [ch - min_ch for ch in channels]
        ref_block = ref_segment[:, channel_indices]
        
        # 计算权重（基于噪声方差）
        noise_var_full = calc_noise_variance(ref_data, fs, start_time, end_time)
        noise_var_group = noise_var_full[channel_indices]
        weights = compute_weights(noise_var_group)
        
        # 加权组合信号
        combined = weighted_combine(ref_block, weights)
        
        # 计算RMS序列
        rms_values = compute_rms(combined, window_size)
        
        # 统计特征
        rms_mean = np.mean(rms_values)
        rms_std = np.std(rms_values)
        rms_min = np.min(rms_values)
        rms_max = np.max(rms_values)
        rms_median = np.median(rms_values)
        rms_p95 = np.percentile(rms_values, 95)
        rms_p99 = np.percentile(rms_values, 99)
        
        threshold_3sigma = rms_mean + 3 * rms_std
        threshold_5sigma = rms_mean + 5 * rms_std
        
        # 判断属于哪个范围
        range_idx, range_info = get_channel_range_for_group(channels, channel_ranges)
        
        all_results.append({
            'group_idx': g_idx + 1,
            'channel_group': f"Ch{channels[0]}-{channels[-1]}",
            'channels': channels,
            'range_idx': range_idx,
            'range_info': range_info,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'rms_min': rms_min,
            'rms_max': rms_max,
            'rms_median': rms_median,
            'rms_p95': rms_p95,
            'rms_p99': rms_p99,
            'threshold_3sigma': threshold_3sigma,
            'threshold_5sigma': threshold_5sigma,
            'num_windows': len(rms_values),
            'weights': weights
        })
    
    # 保存结果
    if group_by_range:
        # 按范围分组保存
        save_manual_results_by_range(all_results, channel_ranges, output_dir)
    else:
        # 整体保存
        save_manual_results_unified(all_results, output_dir)
    
    # 生成总结报告
    generate_manual_summary_report(all_results, group_by_range, channel_ranges, output_dir)
    
    return all_results


def save_manual_results_unified(results, output_dir):
    """保存手动分组结果（统一输出）"""
    
    df = pd.DataFrame([
        {
            'Group_Index': r['group_idx'],
            'Channel_Group': r['channel_group'],
            'Channels': str(r['channels']),
            'Range_Index': r['range_idx'] if r['range_idx'] > 0 else 'N/A',
            'RMS_Mean': r['rms_mean'],
            'RMS_Std': r['rms_std'],
            'RMS_Min': r['rms_min'],
            'RMS_Max': r['rms_max'],
            'RMS_Median': r['rms_median'],
            'RMS_P95': r['rms_p95'],
            'RMS_P99': r['rms_p99'],
            'Threshold_3Sigma': r['threshold_3sigma'],
            'Threshold_5Sigma': r['threshold_5sigma'],
            'Num_Windows': r['num_windows'],
            'Weights': str([f"{w:.4f}" for w in r['weights']])
        }
        for r in results
    ])
    
    csv_path = os.path.join(output_dir, "manual_groups_rms_stats.csv")
    df.to_csv(csv_path, index=False, float_format='%.8f')
    log.info(f"\n✓ 手动分组统计结果已保存：{csv_path}")
    
    # 打印统计信息
    log.info(f"\n{'='*70}")
    log.info(f"手动分组统计摘要")
    log.info(f"{'='*70}")
    log.info(f"通道组数量: {len(results)}")
    log.info(f"\nRMS 均值统计:")
    log.info(f"  最小值: {df['RMS_Mean'].min():.8f}")
    log.info(f"  最大值: {df['RMS_Mean'].max():.8f}")
    log.info(f"  平均值: {df['RMS_Mean'].mean():.8f}")
    log.info(f"  中位数: {df['RMS_Mean'].median():.8f}")
    
    log.info(f"\n建议阈值 (3σ):")
    log.info(f"  最小值: {df['Threshold_3Sigma'].min():.8f}")
    log.info(f"  最大值: {df['Threshold_3Sigma'].max():.8f}")
    log.info(f"  平均值: {df['Threshold_3Sigma'].mean():.8f}")
    log.info(f"  中位数: {df['Threshold_3Sigma'].median():.8f}")


def save_manual_results_by_range(results, channel_ranges, output_dir):
    """保存手动分组结果（按范围分类）"""
    
    # 按范围分组
    range_groups = {}
    unassigned = []
    
    for r in results:
        range_idx = r['range_idx']
        if range_idx > 0:
            if range_idx not in range_groups:
                range_groups[range_idx] = []
            range_groups[range_idx].append(r)
        else:
            unassigned.append(r)
    
    # 保存各范围结果
    for range_idx, group_results in range_groups.items():
        channel_range = channel_ranges[range_idx - 1]
        
        df = pd.DataFrame([
            {
                'Group_Index': r['group_idx'],
                'Channel_Group': r['channel_group'],
                'Channels': str(r['channels']),
                'RMS_Mean': r['rms_mean'],
                'RMS_Std': r['rms_std'],
                'RMS_Min': r['rms_min'],
                'RMS_Max': r['rms_max'],
                'RMS_Median': r['rms_median'],
                'RMS_P95': r['rms_p95'],
                'RMS_P99': r['rms_p99'],
                'Threshold_3Sigma': r['threshold_3sigma'],
                'Threshold_5Sigma': r['threshold_5sigma'],
                'Num_Windows': r['num_windows'],
                'Weights': str([f"{w:.4f}" for w in r['weights']])
            }
            for r in group_results
        ])
        
        csv_path = os.path.join(output_dir,
            f"manual_groups_range{range_idx}_ch{channel_range[0]}-{channel_range[1]}.csv")
        df.to_csv(csv_path, index=False, float_format='%.8f')
        log.info(f"✓ 范围 {range_idx} 结果已保存：{csv_path}")
    
    # 保存未分配的组
    if unassigned:
        df = pd.DataFrame([
            {
                'Group_Index': r['group_idx'],
                'Channel_Group': r['channel_group'],
                'Channels': str(r['channels']),
                'RMS_Mean': r['rms_mean'],
                'RMS_Std': r['rms_std'],
                'Threshold_3Sigma': r['threshold_3sigma'],
                'Threshold_5Sigma': r['threshold_5sigma'],
            }
            for r in unassigned
        ])
        csv_path = os.path.join(output_dir, "manual_groups_unassigned.csv")
        df.to_csv(csv_path, index=False, float_format='%.8f')
        log.info(f"✓ 未分配组结果已保存：{csv_path}")


def generate_manual_summary_report(results, group_by_range, channel_ranges, output_dir):
    """生成手动分组的总结报告"""
    
    log.info(f"\n{'='*70}")
    log.info(f"手动分组 - 总体统计摘要")
    log.info(f"{'='*70}")
    
    rms_means = [r['rms_mean'] for r in results]
    threshold_3sigma = [r['threshold_3sigma'] for r in results]
    threshold_5sigma = [r['threshold_5sigma'] for r in results]
    
    log.info(f"\n全局建议阈值:")
    log.info(f"使用 3σ 策略:")
    log.info(f"  保守值（最小）: {np.min(threshold_3sigma):.8f}")
    log.info(f"  推荐值（中位数）: {np.median(threshold_3sigma):.8f}")
    log.info(f"  激进值（最大）: {np.max(threshold_3sigma):.8f}")
    
    log.info(f"\n使用 5σ 策略:")
    log.info(f"  保守值（最小）: {np.min(threshold_5sigma):.8f}")
    log.info(f"  推荐值（中位数）: {np.median(threshold_5sigma):.8f}")
    log.info(f"  激进值（最大）: {np.max(threshold_5sigma):.8f}")
    
    # 各组详细信息
    log.info(f"\n各通道组详细信息:")
    log.info(f"{'组号':<6} {'通道组':<20} {'均值':<12} {'3σ阈值':<12} {'5σ阈值':<12}")
    log.info(f"{'-'*62}")
    for r in results:
        log.info(f"{r['group_idx']:<6} {r['channel_group']:<20} {r['rms_mean']:<12.8f} "
                f"{r['threshold_3sigma']:<12.8f} {r['threshold_5sigma']:<12.8f}")
    
    # 保存总结CSV
    summary_data = [{
        'Metric': 'Total_Groups',
        'Value': len(results)
    }, {
        'Metric': 'Threshold_3Sigma_Min',
        'Value': np.min(threshold_3sigma)
    }, {
        'Metric': 'Threshold_3Sigma_Median',
        'Value': np.median(threshold_3sigma)
    }, {
        'Metric': 'Threshold_3Sigma_Max',
        'Value': np.max(threshold_3sigma)
    }, {
        'Metric': 'Threshold_5Sigma_Min',
        'Value': np.min(threshold_5sigma)
    }, {
        'Metric': 'Threshold_5Sigma_Median',
        'Value': np.median(threshold_5sigma)
    }, {
        'Metric': 'Threshold_5Sigma_Max',
        'Value': np.max(threshold_5sigma)
    }]
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "manual_groups_threshold_summary.csv")
    df_summary.to_csv(summary_path, index=False, float_format='%.8f')
    log.info(f"\n✓ 总结报告已保存：{summary_path}")


# ================= 自动分组RMS统计分析（原代码）=================
def calculate_background_rms_statistics(ref_file, channel_ranges, channel_step,
                                       num_channels_per_group, fs, lowcut, highcut,
                                       order, start_time, end_time, window_size_sec,
                                       dataset_name, output_dir):
    """计算背景RMS统计特征（自动分组模式）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_range_results = []
    
    for range_idx, channel_range in enumerate(channel_ranges, 1):
        log.info(f"\n{'='*70}")
        log.info(f"处理通道范围 {range_idx}: {channel_range}")
        log.info(f"{'='*70}")
        
        start_channel, end_channel = channel_range
        
        # 生成通道组
        channel_groups = generate_channel_groups(channel_range, channel_step,
                                                num_channels_per_group)
        log.info(f"生成 {len(channel_groups)} 个通道组")
        
        # 加载参考背景数据
        log.info(f"加载背景数据...")
        ref_data = load_hdf5_data(ref_file, dataset_name, channel_range)
        if ref_data is None:
            log.error(f"跳过范围 {range_idx}")
            continue
        
        # 滤波
        log.info(f"应用带通滤波 [{lowcut}-{highcut}] Hz...")
        ref_data = bandpass_filter(ref_data, fs, lowcut, highcut, order)
        
        # 提取分析时间段
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        ref_segment = ref_data[start_idx:end_idx, :]
        log.info(f"分析时间段: {start_time}-{end_time} 秒 ({ref_segment.shape[0]} 个样本点)")
        
        # 计算每个通道组的RMS统计
        window_size = int(window_size_sec * fs)
        
        group_results = []
        
        for g_idx, channels in enumerate(channel_groups):
            # 获取通道索引
            channel_indices = [ch - start_channel for ch in channels]
            ref_block = ref_segment[:, channel_indices]
            
            # 计算权重（基于噪声方差）
            noise_var = calc_noise_variance(ref_data, fs, start_time, end_time)
            noise_var_group = noise_var[channel_indices]
            weights = compute_weights(noise_var_group)
            
            # 加权组合信号
            combined = weighted_combine(ref_block, weights)
            
            # 计算RMS序列
            rms_values = compute_rms(combined, window_size)
            
            # 统计特征
            rms_mean = np.mean(rms_values)
            rms_std = np.std(rms_values)
            rms_min = np.min(rms_values)
            rms_max = np.max(rms_values)
            rms_median = np.median(rms_values)
            rms_p95 = np.percentile(rms_values, 95)
            rms_p99 = np.percentile(rms_values, 99)
            
            # 建议阈值
            threshold_3sigma = rms_mean + 3 * rms_std
            threshold_5sigma = rms_mean + 5 * rms_std
            
            group_results.append({
                'channel_group': f"Ch{channels[0]}-{channels[-1]}",
                'channels': channels,
                'rms_mean': rms_mean,
                'rms_std': rms_std,
                'rms_min': rms_min,
                'rms_max': rms_max,
                'rms_median': rms_median,
                'rms_p95': rms_p95,
                'rms_p99': rms_p99,
                'threshold_3sigma': threshold_3sigma,
                'threshold_5sigma': threshold_5sigma,
                'num_windows': len(rms_values),
                'weights': weights
            })
        
        # 保存该范围的结果
        save_range_results(group_results, channel_range, range_idx, output_dir)
        
        # 汇总统计
        all_range_results.append({
            'range_idx': range_idx,
            'channel_range': channel_range,
            'num_groups': len(channel_groups),
            'results': group_results
        })
    
    # 生成总结报告
    generate_summary_report(all_range_results, output_dir)
    
    return all_range_results


def save_range_results(group_results, channel_range, range_idx, output_dir):
    """保存单个范围的结果"""
    
    # 准备DataFrame
    df = pd.DataFrame([
        {
            'Channel_Group': r['channel_group'],
            'Channels': str(r['channels']),
            'RMS_Mean': r['rms_mean'],
            'RMS_Std': r['rms_std'],
            'RMS_Min': r['rms_min'],
            'RMS_Max': r['rms_max'],
            'RMS_Median': r['rms_median'],
            'RMS_P95': r['rms_p95'],
            'RMS_P99': r['rms_p99'],
            'Threshold_3Sigma': r['threshold_3sigma'],
            'Threshold_5Sigma': r['threshold_5sigma'],
            'Num_Windows': r['num_windows'],
            'Weights': str([f"{w:.4f}" for w in r['weights']])
        }
        for r in group_results
    ])
    
    # 保存CSV
    csv_path = os.path.join(output_dir,
        f"background_rms_stats_range{range_idx}_ch{channel_range[0]}-{channel_range[1]}.csv")
    df.to_csv(csv_path, index=False, float_format='%.8f')
    log.info(f"\n✓ 统计结果已保存：{csv_path}")
    
    # 打印关键统计信息
    log.info(f"\n{'='*70}")
    log.info(f"范围 {range_idx} 统计摘要 (通道 {channel_range[0]}-{channel_range[1]})")
    log.info(f"{'='*70}")
    log.info(f"通道组数量: {len(group_results)}")
    log.info(f"\nRMS 均值统计:")
    log.info(f"  最小值: {df['RMS_Mean'].min():.8f}")
    log.info(f"  最大值: {df['RMS_Mean'].max():.8f}")
    log.info(f"  平均值: {df['RMS_Mean'].mean():.8f}")
    log.info(f"  中位数: {df['RMS_Mean'].median():.8f}")
    
    log.info(f"\nRMS 标准差统计:")
    log.info(f"  最小值: {df['RMS_Std'].min():.8f}")
    log.info(f"  最大值: {df['RMS_Std'].max():.8f}")
    log.info(f"  平均值: {df['RMS_Std'].mean():.8f}")
    
    log.info(f"\n建议阈值 (3σ):")
    log.info(f"  最小值: {df['Threshold_3Sigma'].min():.8f}")
    log.info(f"  最大值: {df['Threshold_3Sigma'].max():.8f}")
    log.info(f"  平均值: {df['Threshold_3Sigma'].mean():.8f}")
    log.info(f"  中位数: {df['Threshold_3Sigma'].median():.8f}")
    
    log.info(f"\n建议阈值 (5σ):")
    log.info(f"  最小值: {df['Threshold_5Sigma'].min():.8f}")
    log.info(f"  最大值: {df['Threshold_5Sigma'].max():.8f}")
    log.info(f"  平均值: {df['Threshold_5Sigma'].mean():.8f}")
    log.info(f"  中位数: {df['Threshold_5Sigma'].median():.8f}")
    
    # 显示前5个通道组的详细信息
    log.info(f"\n前5个通道组详细信息:")
    log.info(f"{'通道组':<15} {'均值':<12} {'标准差':<12} {'3σ阈值':<12} {'5σ阈值':<12}")
    log.info(f"{'-'*63}")
    for i in range(min(5, len(group_results))):
        r = group_results[i]
        log.info(f"{r['channel_group']:<15} {r['rms_mean']:<12.8f} {r['rms_std']:<12.8f} "
                f"{r['threshold_3sigma']:<12.8f} {r['threshold_5sigma']:<12.8f}")


def generate_summary_report(all_range_results, output_dir):
    """生成总结报告"""
    
    log.info(f"\n{'='*70}")
    log.info(f"总体统计摘要")
    log.info(f"{'='*70}")
    
    summary_data = []
    
    for range_result in all_range_results:
        range_idx = range_result['range_idx']
        channel_range = range_result['channel_range']
        results = range_result['results']
        
        rms_means = [r['rms_mean'] for r in results]
        rms_stds = [r['rms_std'] for r in results]
        threshold_3sigma = [r['threshold_3sigma'] for r in results]
        threshold_5sigma = [r['threshold_5sigma'] for r in results]
        
        summary_data.append({
            'Range': f"Range{range_idx}",
            'Channels': f"{channel_range[0]}-{channel_range[1]}",
            'Num_Groups': len(results),
            'Mean_RMS_Mean': np.mean(rms_means),
            'Mean_RMS_Std': np.mean(rms_stds),
            'Min_RMS_Mean': np.min(rms_means),
            'Max_RMS_Mean': np.max(rms_means),
            'Recommended_Threshold_3Sigma': np.median(threshold_3sigma),
            'Recommended_Threshold_5Sigma': np.median(threshold_5sigma),
            'Conservative_Threshold_3Sigma': np.min(threshold_3sigma),
            'Aggressive_Threshold_3Sigma': np.max(threshold_3sigma)
        })
    
    # 保存总结报告
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "threshold_summary_report.csv")
    df_summary.to_csv(summary_path, index=False, float_format='%.8f')
    log.info(f"\n✓ 总结报告已保存：{summary_path}")
    
    # 打印总结
    log.info(f"\n各范围推荐阈值:")
    log.info(f"{'范围':<15} {'通道':<15} {'推荐阈值(3σ)':<18} {'推荐阈值(5σ)':<18}")
    log.info(f"{'-'*66}")
    for data in summary_data:
        log.info(f"{data['Range']:<15} {data['Channels']:<15} "
                f"{data['Recommended_Threshold_3Sigma']:<18.8f} "
                f"{data['Recommended_Threshold_5Sigma']:<18.8f}")
    
    # 全局建议
    all_3sigma = [d['Recommended_Threshold_3Sigma'] for d in summary_data]
    all_5sigma = [d['Recommended_Threshold_5Sigma'] for d in summary_data]
    
    log.info(f"\n{'='*70}")
    log.info(f"全局建议阈值")
    log.info(f"{'='*70}")
    log.info(f"使用 3σ 策略:")
    log.info(f"  保守值（最小）: {np.min(all_3sigma):.8f}")
    log.info(f"  推荐值（中位数）: {np.median(all_3sigma):.8f}")
    log.info(f"  激进值（最大）: {np.max(all_3sigma):.8f}")
    
    log.info(f"\n使用 5σ 策略:")
    log.info(f"  保守值（最小）: {np.min(all_5sigma):.8f}")
    log.info(f"  推荐值（中位数）: {np.median(all_5sigma):.8f}")
    log.info(f"  激进值（最大）: {np.max(all_5sigma):.8f}")
    
    log.info(f"\n💡 使用建议:")
    log.info(f"  - 如果希望捕捉更多事件，使用较低阈值（3σ保守值）")
    log.info(f"  - 如果希望减少误报，使用较高阈值（5σ推荐值）")
    log.info(f"  - 原代码使用的阈值: 0.015")
    log.info(f"  - 与原阈值对比: {np.median(all_3sigma)/0.015:.2f}x (3σ) / {np.median(all_5sigma)/0.015:.2f}x (5σ)")


def main():
    log.info("="*70)
    log.info("DAS 背景 RMS 统计分析 - 阈值计算")
    log.info("="*70)
    log.info(f"分组模式: {'自动分组' if grouping_mode == 'auto' else '手动分组'}")
    
    start_time_process = datetime.now()
    
    if grouping_mode == "auto":
        # 自动分组模式
        log.info(f"通道范围: {channel_ranges}")
        log.info(f"滑动步长: {channel_step}, 每组通道数: {num_channels_per_group}")
        
        results = calculate_background_rms_statistics(
            ref_file=ref_file,
            channel_ranges=channel_ranges,
            channel_step=channel_step,
            num_channels_per_group=num_channels_per_group,
            fs=fs,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
            start_time=start_time,
            end_time=end_time,
            window_size_sec=window_size_sec,
            dataset_name=dataset_name,
            output_dir=output_dir
        )
    
    elif grouping_mode == "manual":
        # 手动分组模式
        log.info(f"手动指定通道组数量: {len(manual_channel_groups)}")
        log.info(f"按范围分类输出: {manual_group_by_range}")
        
        results = calculate_manual_group_rms_statistics(
            ref_file=ref_file,
            manual_groups=manual_channel_groups,
            channel_ranges=channel_ranges,
            group_by_range=manual_group_by_range,
            fs=fs,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
            start_time=start_time,
            end_time=end_time,
            window_size_sec=window_size_sec,
            dataset_name=dataset_name,
            output_dir=output_dir
        )
    
    else:
        log.error(f"未知的分组模式: {grouping_mode}，请使用 'auto' 或 'manual'")
        return
    
    duration = (datetime.now() - start_time_process).total_seconds()
    log.info(f"\n{'='*70}")
    log.info(f"分析完成！耗时: {duration:.2f} 秒")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
