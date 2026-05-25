# -*- coding: utf-8 -*-
"""
DAS 应变分析可视化工具 - 增强版 (含CSV导出 + 自定义色阶范围)

功能：
1. [新增] 将读取的应变数据按整秒时间窗口对齐，导出为CSV表格
2. [新增] 支持提取所有通道到CSV（此时不绘制图像）
3. 绘制应变率和应变的热力图（支持自定义色阶范围）
4. 绘制应变率和应变的曲线图（单列堆叠，分开导出）

@author: Claude (Modified)
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from glob import glob
import gc
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("DAS-Viz")


# ==================== 配置区域 ====================

# --- 数据目录 ---
data_dir = r""

# --- 输出目录 ---
output_dir = r""

# --- 时间配置 ---
start_datetime_str = "2025/07/07/14:06:54"
datetime_format = "%Y/%m/%d/%H:%M:%S"
file_duration = 600.0  # 每个文件时长（秒）
time_label_interval_hours = 1.0  # 绘图时间标签间隔（小时）

# --- 降采样因子 (读取时) ---
# 注意：如果原始采样率很高（如2000Hz），这里设为2000意味着读取进来大概是1Hz。
# 如果你想导出的CSV更精确，确保读取进来的数据频率 >= 1Hz。
downsample_factor = 2000

# --- CSV 导出配置 ---
export_csv = True             # 是否导出 CSV
csv_filename = "240-257strain_data.csv" # 输出文件名

# --- [新增] 是否提取所有通道 ---
# True  = 提取所有通道到CSV，不绘制图像
# False = 只提取 plot_group_indices 指定的通道，并绘制图像
extract_all_channels = False

# --- 选择要绘制/导出的通道组索引 ---
# 当 extract_all_channels = False 时生效
plot_group_indices = [32, 33, 34, 35]

# --- 选择文件范围 ---
# None = 全部
file_range = None

# ==================== 图片配置 ====================

fig_width = 8.0
fig_height_per_panel = 1.8
output_format = 'png'
output_dpi = 600

font_family = 'Times New Roman'
font_size_title = 12
font_size_label = 10
font_size_tick = 9
font_weight_title = 'bold'

line_width = 1.0

heatmap_cmap = 'RdBu_r'
heatmap_symmetric = True

# --- 热力图色阶自定义范围 (新增功能) ---
# 设为 None 则自动计算（使用99百分位），设为具体数值则使用该值
# 示例: heatmap_strain_rate_vmin = -0.001, heatmap_strain_rate_vmax = 0.001
heatmap_strain_rate_vmin = None  # 应变率色阶最小值
heatmap_strain_rate_vmax = None  # 应变率色阶最大值
heatmap_strain_vmin = None       # 累积应变色阶最小值
heatmap_strain_vmax = None       # 累积应变色阶最大值

curve_colorscheme = 'strain_classic'

CURVE_COLORS = {
    'strain_classic': [
        '#D62728', '#1F77B4', '#2CA02C', '#FF7F0E',
        '#9467BD', '#8C564B', '#E377C2', '#17BECF',
        '#BCBD22', '#7F7F7F'
    ]
}

# ==================== 代码实现 ====================


def setup_style():
    """设置绘图样式"""
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if font_family in available_fonts:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_family, 'DejaVu Serif']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
    
    plt.rcParams['font.size'] = font_size_tick
    plt.rcParams['axes.titlesize'] = font_size_title
    plt.rcParams['axes.labelsize'] = font_size_label
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = output_dpi
    plt.rcParams['savefig.bbox'] = 'tight'


class StrainDataReader:
    """应变数据读取器"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._scan_files()
    
    def _scan_files(self):
        pattern = os.path.join(self.data_dir, 'strain_*.h5')
        self.file_list = sorted([os.path.basename(f) for f in glob(pattern)])
        self.num_files = len(self.file_list)
        
        if self.num_files == 0:
            raise RuntimeError(f"未找到文件: {pattern}")
        
        first_file = os.path.join(self.data_dir, self.file_list[0])
        with h5py.File(first_file, 'r') as f:
            self.fs = f.attrs['fs']
            self.num_groups = f.attrs['num_groups']
            self.group_labels = [
                l.decode() if isinstance(l, bytes) else l 
                for l in f.attrs['group_labels']
            ]
        
        log.info(f"找到 {self.num_files} 个文件, {self.num_groups} 个通道组")
    
    def load_data(self, group_indices=None, downsample=1, file_range=None):
        if group_indices is None:
            group_indices = list(range(self.num_groups))
        
        if file_range is None:
            file_range = (0, self.num_files)
        
        start_file, end_file = file_range
        end_file = min(end_file, self.num_files)
        files_to_load = self.file_list[start_file:end_file]
        
        log.info(f"读取 {len(files_to_load)} 个文件 (Downsample={downsample})")
        log.info(f"提取通道数: {len(group_indices)}")
        
        all_strain_rate = []
        all_strain = []
        
        for i, filename in enumerate(files_to_load):
            if (i + 1) % 10 == 0:
                log.info(f"  进度: {i+1}/{len(files_to_load)}")
            
            filepath = os.path.join(self.data_dir, filename)
            
            with h5py.File(filepath, 'r') as f:
                keys = list(f.keys())
                strain_rate_key = [k for k in keys if 'strain_rate' in k][0]
                strain_key = [k for k in keys if 'strain' in k and 'rate' not in k][0]
                
                num_samples = f[strain_key].shape[0]
                indices = np.arange(0, num_samples, downsample)
                
                sr = f[strain_rate_key][indices, :][:, group_indices].astype(np.float32)
                s = f[strain_key][indices, :][:, group_indices].astype(np.float32)
                
                all_strain_rate.append(sr)
                all_strain.append(s)
        
        strain_rate = np.concatenate(all_strain_rate, axis=0)
        strain = np.concatenate(all_strain, axis=0)
        labels = [self.group_labels[i] for i in group_indices]
        
        del all_strain_rate, all_strain
        gc.collect()
        
        return {
            'strain_rate': strain_rate,
            'strain': strain,
            'labels': labels,
            'num_files': len(files_to_load),
            'fs': self.fs,
            'downsample': downsample
        }


def save_to_csv(data, start_datetime_str, datetime_format, file_duration, output_dir):
    """
    将应变数据保存为 CSV（不做重采样和插值，保留真实时间和原始值）
    """
    log.info("正在处理 CSV 导出数据...")
    
    start_datetime = datetime.strptime(start_datetime_str, datetime_format)
    strain = data['strain']  # 默认导出累积应变，如需应变率改用 strain_rate
    labels = data['labels']
    num_files = data['num_files']
    downsample = data['downsample']
    original_fs = data['fs']
    
    num_samples, num_channels = strain.shape
    
    # 1. 计算读取后数据的实际采样率和时间间隔
    current_fs = original_fs / downsample
    total_seconds = num_files * file_duration
    dt = total_seconds / num_samples  # 与绘图逻辑一致
    
    log.info(f"原始采样率: {original_fs} Hz, 降采样因子: {downsample}")
    log.info(f"当前数据实际采样率: {current_fs:.4f} Hz")
    log.info(f"数据点间隔: {dt:.4f} 秒")
    log.info(f"导出通道数: {num_channels}, 数据点数: {num_samples}")
    
    # 2. 构建真实时间索引（与绘图逻辑一致：linspace）
    time_sec = np.linspace(0, total_seconds, num_samples)
    time_index = [start_datetime + timedelta(seconds=s) for s in time_sec]

    # 3. 创建 DataFrame（不做任何重采样或插值）
    df = pd.DataFrame(data=strain, index=time_index, columns=labels)
    df.index.name = 'Time'
    
    # 4. 直接保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, csv_filename)
        
        # 保存：时间格式精确到毫秒
        df.to_csv(csv_path, date_format='%Y/%m/%d %H:%M:%S.%f')
        
        log.info(f"CSV 已保存: {csv_path}")
        log.info(f"CSV 数据形状: {df.shape}")
        log.info(f"CSV 时间范围: {df.index[0]} -> {df.index[-1]}")
    else:
        log.warning("未指定输出目录，跳过保存 CSV")


def plot_heatmap_full(data, start_datetime_str, datetime_format, file_duration,
                 time_label_interval_hours, cmap, symmetric,
                 output_dir, show=True,
                 strain_rate_vmin=None, strain_rate_vmax=None,
                 strain_vmin=None, strain_vmax=None):
    """绘制完整热力图
    
    参数:
        data: 数据字典
        start_datetime_str: 起始时间字符串
        datetime_format: 时间格式
        file_duration: 每个文件时长（秒）
        time_label_interval_hours: 时间标签间隔（小时）
        cmap: 颜色映射
        symmetric: 是否对称色阶
        output_dir: 输出目录
        show: 是否显示图像
        strain_rate_vmin: 应变率色阶最小值，None则自动计算
        strain_rate_vmax: 应变率色阶最大值，None则自动计算
        strain_vmin: 累积应变色阶最小值，None则自动计算
        strain_vmax: 累积应变色阶最大值，None则自动计算
    """
    start_datetime = datetime.strptime(start_datetime_str, datetime_format)
    strain_rate = data['strain_rate']
    strain = data['strain']
    labels = data['labels']
    num_files = data['num_files']
    
    num_samples, num_channels = strain.shape
    total_seconds = num_files * file_duration
    total_hours = total_seconds / 3600
    
    fig_height = fig_height_per_panel * 2 + 1
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    extent = [0, total_hours, num_channels - 0.5, -0.5]
    
    # ===== Strain Rate 热力图 =====
    ax = axes[0]
    data_matrix = strain_rate.T
    
    # 计算或使用自定义色阶范围
    if strain_rate_vmax is not None:
        vmax = strain_rate_vmax
    else:
        vmax = np.percentile(np.abs(data_matrix), 99)
    
    if strain_rate_vmin is not None:
        vmin = strain_rate_vmin
    elif symmetric:
        vmin = -vmax
    else:
        vmin = np.percentile(data_matrix, 1)
    
    log.info(f"Strain Rate 色阶范围: [{vmin:.6f}, {vmax:.6f}]")
    
    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Strain rate')
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels(labels)
    ax.set_title('(a) Strain rate', loc='left')
    ax.set_xticklabels([])
    
    # ===== Cumulative Strain 热力图 =====
    ax = axes[1]
    data_matrix = strain.T
    
    # 计算或使用自定义色阶范围
    if strain_vmax is not None:
        vmax = strain_vmax
    else:
        vmax = np.percentile(np.abs(data_matrix), 99)
    
    if strain_vmin is not None:
        vmin = strain_vmin
    elif symmetric:
        vmin = -vmax
    else:
        vmin = np.percentile(data_matrix, 1)
    
    log.info(f"Cumulative Strain 色阶范围: [{vmin:.6f}, {vmax:.6f}]")
    
    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Strain')
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels(labels)
    ax.set_title('(b) Cumulative strain', loc='left')
    
    # ===== Time Axis =====
    end_datetime = start_datetime + timedelta(seconds=total_seconds)
    crosses_midnight = start_datetime.date() != end_datetime.date()
    tick_pos = np.arange(0, total_hours + 0.01, time_label_interval_hours)
    ax.set_xticks(tick_pos)
    
    tick_labels = []
    for h in tick_pos:
        dt = start_datetime + timedelta(hours=h)
        if crosses_midnight:
            tick_labels.append(dt.strftime('%m-%d %H:%M'))
        else:
            tick_labels.append(dt.strftime('%H:%M'))
            
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    if crosses_midnight:
        ax.set_xlabel('Time')
    else:
        ax.set_xlabel(f'Time ({start_datetime.strftime("%Y-%m-%d")})')
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'strain_heatmap.{output_format}'), dpi=output_dpi)
        log.info(f"热力图已保存")
    if show:
        plt.show()
    else:
        plt.close()


def plot_curves(data, start_datetime_str, datetime_format, file_duration,
                time_label_interval_hours, output_dir, show=True):
    """绘制堆叠曲线图"""
    start_datetime = datetime.strptime(start_datetime_str, datetime_format)
    strain_rate = data['strain_rate']
    strain = data['strain']
    labels = data['labels']
    num_files = data['num_files']
    num_samples, num_channels = strain.shape
    
    total_seconds = num_files * file_duration
    time_sec = np.linspace(0, total_seconds, num_samples)
    time_datetime = [start_datetime + timedelta(seconds=s) for s in time_sec]
    
    colors = CURVE_COLORS.get(curve_colorscheme, CURVE_COLORS['strain_classic'])
    if num_channels > len(colors):
        colors = colors * (num_channels // len(colors) + 1)

    def _draw_stack(data_matrix, title, suffix):
        h = fig_height_per_panel * num_channels + 0.8
        fig, axes = plt.subplots(num_channels, 1, figsize=(fig_width, h), sharex=True, squeeze=False)
        for i in range(num_channels):
            ax = axes[i, 0]
            c = colors[i % len(colors)]
            ax.plot(time_datetime, data_matrix[:, i], color=c, lw=line_width)
            ax.set_ylabel(labels[i], color=c, fontweight='bold')
            ax.grid(True, alpha=0.3, ls='--')
            if i == 0:
                ax.set_title(title, loc='left', fontweight='bold')
            
        ax_btm = axes[-1, 0]
        end_datetime = start_datetime + timedelta(seconds=total_seconds)
        crosses_midnight = start_datetime.date() != end_datetime.date()
        
        ax_btm.xaxis.set_major_locator(mdates.HourLocator(interval=int(time_label_interval_hours)))
        
        if crosses_midnight:
            ax_btm.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax_btm.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
             
        plt.setp(ax_btm.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_btm.set_xlabel('Time')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'strain_{suffix}_stacked.{output_format}'), dpi=output_dpi)
            log.info(f"{suffix} 曲线图已保存")
        if show:
            plt.show()
        else:
            plt.close()

    log.info("绘制应变率堆叠图...")
    _draw_stack(strain_rate, "(a) Strain Rate", "rate")
    log.info("绘制应变堆叠图...")
    _draw_stack(strain, "(b) Cumulative Strain", "cumulative")


def main():
    log.info("=" * 50)
    log.info("DAS 数据处理与导出工具")
    log.info("=" * 50)
    
    setup_style()
    
    reader = StrainDataReader(data_dir)
    
    # 根据 extract_all_channels 决定提取哪些通道
    if extract_all_channels:
        # 提取所有通道
        log.info("模式: 提取所有通道到CSV (不绘制图像)")
        group_indices = None  # None 表示所有通道
    else:
        # 只提取指定通道
        log.info(f"模式: 提取指定通道 {plot_group_indices} (导出CSV + 绘制图像)")
        group_indices = plot_group_indices
    
    data = reader.load_data(
        group_indices=group_indices,
        downsample=downsample_factor,
        file_range=file_range
    )
    
    # 1. 导出 CSV
    if export_csv:
        save_to_csv(
            data,
            start_datetime_str=start_datetime_str,
            datetime_format=datetime_format,
            file_duration=file_duration,
            output_dir=output_dir
        )
    
    # 2. 绘制图像（仅在非全部通道模式下执行）
    if not extract_all_channels:
        # 绘制热力图（支持自定义色阶范围）
        plot_heatmap_full(
            data,
            start_datetime_str=start_datetime_str,
            datetime_format=datetime_format,
            file_duration=file_duration,
            time_label_interval_hours=time_label_interval_hours,
            cmap=heatmap_cmap,
            symmetric=heatmap_symmetric,
            output_dir=output_dir,
            show=True,
            strain_rate_vmin=heatmap_strain_rate_vmin,
            strain_rate_vmax=heatmap_strain_rate_vmax,
            strain_vmin=heatmap_strain_vmin,
            strain_vmax=heatmap_strain_vmax
        )
        
        # 绘制曲线图
        plot_curves(
            data,
            start_datetime_str=start_datetime_str,
            datetime_format=datetime_format,
            file_duration=file_duration,
            time_label_interval_hours=time_label_interval_hours,
            output_dir=output_dir,
            show=True
        )
    else:
        log.info("已跳过图像绘制（全部通道模式）")
    
    log.info("全部任务完成!")


if __name__ == "__main__":
    main()
