# -*- coding: utf-8 -*-
"""
计算并绘制 P90 连通损伤核组织化指标

输出内容：
1. core_organization_P90_table.csv
2. core_organization_P90_mainfig.png
   - N90(t)
   - Lmax90(t)，log 坐标
   - D90(t)
3. Lmax90_and_D90_curve_LOG.png
   - Lmax90(t)，log 坐标
   - D90(t)

核心指标：
N90        = P90 阈值下的连通损伤核数量
Lmax90     = P90 阈值下最大连通损伤核体积
D90        = Lmax90 / total_high90
total_high90 = P90 阈值以上所有高损伤体素总数
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage


# ============================================================
# 1. 用户配置区：你主要改这里
# ============================================================

CSV_FILES = [
    # (路径, 工况, 时刻标签, 降雨后小时数)

    (r'', 'S1', 'T_17:34', 3.56),
    (r'', 'S1', 'T_18:12', 4.20),
    (r'', 'S1', 'T_19:34', 5.56),
    (r'', 'S1', 'T_21:00', 7.00),
    (r'', 'S1', 'T_21:42', 7.70),
    (r'', 'S1', 'T_22:08', 8.13),
    (r'', 'S1', 'T_22:38', 8.63),

    (r'', 'S2', 'T_10:55', 1.91),
    (r'', 'S2', 'T_12:00', 3.00),
    (r'', 'S2', 'T_15:13', 6.21),
    (r'', 'S2', 'T_18:06', 9.10),
    (r'', 'S2', 'T_20:37', 11.61),

    (r'', 'S4', 'T_12:00', 3.00),
    (r'', 'S4', 'T_16:00', 7.00),
    (r'', 'S4', 'T_20:00', 11.00),
    (r'', 'S4', 'T_20:30', 11.50),
]

# 各工况失稳时刻
FAILURE_HOURS = {
    'S1': 8.63,
    'S2': 11.61,
    'S4': 11.50,
}

# 输出文件夹
OUTPUT_DIR = r''

# 坡体几何参数，保持和你原脚本一致
SLOPE_CREST_LENGTH = 100
SLOPE_TOE_LENGTH = 180
SLOPE_HEIGHT = 100

# 主分析阈值
MAIN_PERCENTILE = 90

# 工况颜色
CASE_COLORS = {
    'S1': '#d62728',
    'S2': '#1f77b4',
    'S4': '#2ca02c',
}


# ============================================================
# 2. 数据读取与坡体 mask
# ============================================================

def load_voxler_csv(path):
    """
    读取 Voxler 导出的体素 CSV。
    要求 CSV 为四列：x, y, z, v，无表头。
    """
    df = pd.read_csv(path, header=None, names=['x', 'y', 'z', 'v'])

    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    zs = np.sort(df['z'].unique())

    nx, ny, nz = len(xs), len(ys), len(zs)

    if nx * ny * nz != len(df):
        raise ValueError(
            f"网格尺寸不匹配：nx*ny*nz={nx*ny*nz}, 但 CSV 行数={len(df)}"
        )

    df_sorted = df.sort_values(['x', 'y', 'z']).reset_index(drop=True)
    V = df_sorted['v'].values.reshape(nx, ny, nz)

    return V, xs, ys, zs


def build_slope_mask(xs, ys, zs):
    """
    构建坡体内部 mask。
    """
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')

    slope_dx = SLOPE_TOE_LENGTH - SLOPE_CREST_LENGTH
    x_max_at_z = SLOPE_TOE_LENGTH - (slope_dx / SLOPE_HEIGHT) * Z3

    mask = (
        (Z3 >= 0) &
        (Z3 <= SLOPE_HEIGHT) &
        (X3 >= 0) &
        (X3 <= x_max_at_z)
    )

    return mask, X3, Y3, Z3


# ============================================================
# 3. 全局 P90 阈值
# ============================================================

def compute_global_threshold(csv_files, percentile=90):
    """
    计算所有工况、所有时刻共同的全局 P90 阈值。

    注意：
    这里必须使用全局阈值。
    不建议每个时刻单独计算 P90，否则不同时间之间不可直接比较。
    """
    all_inside_values = []

    for path, case, label, hours in csv_files:
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在：{path}")
            continue

        V, xs, ys, zs = load_voxler_csv(path)
        slope_mask, _, _, _ = build_slope_mask(xs, ys, zs)

        inside_values = V[slope_mask]
        all_inside_values.append(inside_values)

    if len(all_inside_values) == 0:
        raise RuntimeError("没有成功读取任何 CSV 文件。请检查 CSV_FILES 路径。")

    all_inside_values = np.concatenate(all_inside_values)
    threshold = float(np.percentile(all_inside_values, percentile))

    return threshold


# ============================================================
# 4. 单时刻连通核分析
# ============================================================

def analyze_one_snapshot(path, case, label, hours, threshold):
    """
    对单个时刻计算：
    N90, Lmax90, total_high90, D90
    """
    V, xs, ys, zs = load_voxler_csv(path)
    slope_mask, X3, Y3, Z3 = build_slope_mask(xs, ys, zs)

    high = (V > threshold) & slope_mask
    total_high_size = int(high.sum())

    # 26 邻域三维连通
    structure = np.ones((3, 3, 3), dtype=int)
    labeled, n_components = ndimage.label(high, structure=structure)

    component_sizes = []

    for cid in range(1, n_components + 1):
        size = int((labeled == cid).sum())
        component_sizes.append(size)

    if len(component_sizes) == 0:
        largest_size = 0
        dominance_D90 = np.nan
    else:
        largest_size = int(np.max(component_sizes))
        dominance_D90 = largest_size / total_high_size if total_high_size > 0 else np.nan

    return {
        'case': case,
        'label': label,
        'hours': hours,
        'threshold_P90': threshold,
        'n_components_90': int(n_components),
        'largest_size_90': int(largest_size),
        'total_high_size_90': int(total_high_size),
        'dominance_D90': float(dominance_D90) if not np.isnan(dominance_D90) else np.nan,
    }


# ============================================================
# 5. 绘图函数
# ============================================================

def prepare_log_y(series):
    """
    log 坐标不能画 0。
    这里把 0 替换为 NaN，图中自动不显示这些点。
    这样比把 0 强行改成 1 更严谨。
    """
    arr = np.asarray(series, dtype=float)
    arr[arr <= 0] = np.nan
    return arr


def plot_core_organization_mainfig(df, output_path):
    """
    三面板主图：
    (a) N90(t)
    (b) Lmax90(t), log 坐标
    (c) D90(t)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    cases = sorted(df['case'].unique())

    for case in cases:
        sub = df[df['case'] == case].sort_values('hours')
        color = CASE_COLORS.get(case, None)

        x = sub['hours'].values
        n90 = sub['n_components_90'].values
        lmax90 = prepare_log_y(sub['largest_size_90'].values)
        d90 = sub['dominance_D90'].values

        # (a) N90
        axes[0].plot(
            x, n90,
            marker='o',
            linewidth=2,
            markersize=6,
            label=case,
            color=color,
        )

        # (b) Lmax90, log 坐标
        axes[1].plot(
            x, lmax90,
            marker='s',
            linewidth=2,
            markersize=6,
            label=case,
            color=color,
        )

        # (c) D90
        axes[2].plot(
            x, d90,
            marker='o',
            linewidth=2,
            markersize=6,
            label=case,
            color=color,
        )

        # 失稳时刻虚线
        if case in FAILURE_HOURS:
            for ax in axes:
                ax.axvline(
                    FAILURE_HOURS[case],
                    linestyle='--',
                    linewidth=1.4,
                    alpha=0.45,
                    color=color,
                )

    # (a)
    axes[0].set_xlabel('Time since rainfall onset (h)')
    axes[0].set_ylabel(r'Number of connected components, $N_{90}$')
    axes[0].set_title('(a) Fragmentation of high-damage voxels')
    axes[0].grid(alpha=0.3)
    axes[0].legend(title='Case')

    # (b)
    axes[1].set_xlabel('Time since rainfall onset (h)')
    axes[1].set_ylabel(r'Largest core size, $L_{\max,90}$ (voxels)')
    axes[1].set_yscale('log')
    axes[1].set_title('(b) Growth of the dominant connected core')
    axes[1].grid(alpha=0.3, which='both')
    axes[1].legend(title='Case')

    # 根据数据自动设定 log 坐标下限
    positive_lmax = df.loc[df['largest_size_90'] > 0, 'largest_size_90']
    if len(positive_lmax) > 0:
        ymin = max(1, positive_lmax.min() * 0.5)
        ymax = positive_lmax.max() * 1.8
        axes[1].set_ylim(ymin, ymax)

    # (c)
    axes[2].set_xlabel('Time since rainfall onset (h)')
    axes[2].set_ylabel(r'Dominance ratio, $D_{90}$')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title('(c) Dominance of the largest connected core')
    axes[2].grid(alpha=0.3)
    axes[2].legend(title='Case')

    fig.suptitle(
        'Quantitative organization of connected damage cores before failure',
        fontsize=14,
        fontweight='bold',
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_Lmax_and_D90_log(df, output_path):
    """
    双面板图：
    上图：Lmax90，log 坐标
    下图：D90，线性坐标
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True)

    cases = sorted(df['case'].unique())

    for case in cases:
        sub = df[df['case'] == case].sort_values('hours')
        color = CASE_COLORS.get(case, None)

        x = sub['hours'].values
        lmax90 = prepare_log_y(sub['largest_size_90'].values)
        d90 = sub['dominance_D90'].values

        axes[0].plot(
            x, lmax90,
            marker='s',
            linewidth=2.2,
            markersize=7,
            label=case,
            color=color,
        )

        axes[1].plot(
            x, d90,
            marker='o',
            linewidth=2.2,
            markersize=7,
            label=case,
            color=color,
        )

        if case in FAILURE_HOURS:
            for ax in axes:
                ax.axvline(
                    FAILURE_HOURS[case],
                    linestyle='--',
                    linewidth=1.6,
                    alpha=0.45,
                    color=color,
                )

    # Lmax90 log 坐标
    axes[0].set_ylabel(r'$L_{\max,90}$ (voxels)')
    axes[0].set_yscale('log')
    axes[0].set_title('Largest connected damage core size')
    axes[0].grid(alpha=0.3, which='both')
    axes[0].legend(title='Case')

    positive_lmax = df.loc[df['largest_size_90'] > 0, 'largest_size_90']
    if len(positive_lmax) > 0:
        ymin = max(1, positive_lmax.min() * 0.5)
        ymax = positive_lmax.max() * 1.8
        axes[0].set_ylim(ymin, ymax)

    # D90
    axes[1].set_xlabel('Time since rainfall onset (h)')
    axes[1].set_ylabel(r'$D_{90}$')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Dominance ratio of the largest connected damage core')
    axes[1].grid(alpha=0.3)
    axes[1].legend(title='Case')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_Lmax_log_only(df, output_path):
    """
    单独输出一张 Lmax90 log 图。
    如果你只想放最大核增长曲线，可以用这张。
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    cases = sorted(df['case'].unique())

    for case in cases:
        sub = df[df['case'] == case].sort_values('hours')
        color = CASE_COLORS.get(case, None)

        x = sub['hours'].values
        lmax90 = prepare_log_y(sub['largest_size_90'].values)

        ax.plot(
            x, lmax90,
            marker='s',
            linewidth=2.2,
            markersize=7,
            label=case,
            color=color,
        )

        if case in FAILURE_HOURS:
            ax.axvline(
                FAILURE_HOURS[case],
                linestyle='--',
                linewidth=1.6,
                alpha=0.45,
                color=color,
            )

    ax.set_xlabel('Time since rainfall onset (h)')
    ax.set_ylabel(r'Largest connected core size, $L_{\max,90}$ (voxels)')
    ax.set_yscale('log')
    ax.set_title(r'Pre-failure growth of $L_{\max,90}$')
    ax.grid(alpha=0.3, which='both')
    ax.legend(title='Case')

    positive_lmax = df.loc[df['largest_size_90'] > 0, 'largest_size_90']
    if len(positive_lmax) > 0:
        ymin = max(1, positive_lmax.min() * 0.5)
        ymax = positive_lmax.max() * 1.8
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("计算全局 P90 阈值")
    print("=" * 80)

    threshold_P90 = compute_global_threshold(
        CSV_FILES,
        percentile=MAIN_PERCENTILE,
    )

    print(f"全局 P90 阈值 = {threshold_P90:.6f}")
    print()

    print("=" * 80)
    print("逐时刻计算 N90, Lmax90, total_high90, D90")
    print("=" * 80)

    rows = []

    for path, case, label, hours in CSV_FILES:
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在：{path}")
            continue

        print(f"正在处理：{case} {label} ({hours:.2f} h)")

        row = analyze_one_snapshot(
            path=path,
            case=case,
            label=label,
            hours=hours,
            threshold=threshold_P90,
        )

        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("没有任何文件被成功处理，请检查 CSV_FILES 路径。")

    df = pd.DataFrame(rows)
    df = df.sort_values(['case', 'hours']).reset_index(drop=True)

    # 输出表格
    table_path = os.path.join(OUTPUT_DIR, 'core_organization_P90_table.csv')
    df.to_csv(table_path, index=False, encoding='utf-8-sig')

    # 输出图片
    mainfig_path = os.path.join(OUTPUT_DIR, 'core_organization_P90_mainfig.png')
    doublefig_path = os.path.join(OUTPUT_DIR, 'Lmax90_and_D90_curve_LOG.png')
    lmax_only_path = os.path.join(OUTPUT_DIR, 'Lmax90_LOG_only.png')

    plot_core_organization_mainfig(df, mainfig_path)
    plot_Lmax_and_D90_log(df, doublefig_path)
    plot_Lmax_log_only(df, lmax_only_path)

    print()
    print("=" * 80)
    print("完成")
    print("=" * 80)
    print(f"结果表：{table_path}")
    print(f"三面板主图：{mainfig_path}")
    print(f"Lmax90 + D90 双面板图：{doublefig_path}")
    print(f"Lmax90 单独 log 图：{lmax_only_path}")
    print()
    print("注意：")
    print("1. Lmax90=0 的点不会显示在 log 图中，这是正常的。")
    print("2. 不要在论文里写严格 exponential growth，除非后续做拟合。")
    print("3. 建议表述为 rapid growth on a logarithmic scale 或 near log-linear growth。")


if __name__ == '__main__':
    main()
