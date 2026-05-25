"""
================================================================================
 DAS 主损伤核增长分析 — 论文图版 v3
================================================================================
 产出文件：

   fig1_Lmax_logistic.pdf/.png
   fig2_logistic_params.pdf/.png
   fig3a_model_heatmaps.pdf/.png   ← 三热图（原fig3左侧）
   fig3b_delta_aicc.pdf/.png       ← ΔAICc柱状图（原fig3右侧）

 字体：Times New Roman 10pt
 宽度基于 A4 正文宽（160mm）

 v3 改动（仅 plot_fig1_lmax）：
   ① Logistic 图例只写 'Logistic fit'
   ② 参数文本框移至左侧空白区
   ③ 指数虚线由 polyfit 从数据拟合（形状真实）
      图例中 λ 值显示 EXP_LAMBDAS 硬编码值（与文中一致）
   ④ Exp. approx. 标签换行，缩短图例宽度

 v3 fig3 改动：
   ⑤ 三热图拆为 fig3a，单独输出，适当加宽
   ⑥ ΔAICc 柱状图拆为 fig3b，单独输出
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage, optimize
from pathlib import Path

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILES = [
    (r'',  'S1', 'T_17:34',  3.56),
    (r'',  'S1', 'T_18:12',  4.20),
    (r'',  'S1', 'T_19:34',  5.56),
    (r'',  'S1', 'T_21:00',  7.00),
    (r'',  'S1', 'T_21:42',  7.70),
    (r'',  'S1', 'T_22:08',  8.13),
    (r'',  'S1', 'T_22:38',  8.63),
    (r'','S2', 'T_10:55',  1.91),
    (r'','S2', 'T_12:00',  3.00),
    (r'','S2', 'T_15:13',  6.21),
    (r'','S2', 'T_18:06',  9.10),
    (r'','S2', 'T_20:37', 11.61),
    (r'',  'S4', 'T_12:00',  3.00),
    (r'',  'S4', 'T_16:00',  7.00),
    (r'',  'S4', 'T_17:00',  8.00),
    (r'',  'S4', 'T_18:30',  9.00),
    (r'',  'S4', 'T_19:00', 10.00),
    # (r'',  'S4', 'T_20:00', 11.00),
    (r'',  'S4', 'T_20:30', 11.50),
]

FAILURE_HOURS = {'S1': 8.63, 'S2': 11.61, 'S4': 11.50}

FIT_WINDOWS = {
    'S1': [3.56, 8.63],
    'S2': [1.91, 11.61],
    'S4': [3.00, 11.50],
}

EXP_WINDOWS = {
    'S1': [7.00, 8.63],
    'S2': [9.10, 11.61],
    'S4': [9.00, 11.50],
}

# 文中计算的 λ 值，仅用于图例标签显示，不参与曲线拟合
EXP_LAMBDAS = {'S1': 0.906, 'S2': 0.772, 'S4': 0.766}

OUTPUT_DIR = r''

SLOPE_CREST_LENGTH    = 100
SLOPE_TOE_LENGTH      = 180
SLOPE_HEIGHT          = 100
MAIN_PCT              = 90
MIN_COMP_VOXELS       = 5
THRESHOLD_PERCENTILES = [85, 90, 95]
BOOTSTRAP_N           = 2000
RANDOM_SEED           = 42

# ─────────────────────────────────────────────────────────────────────────────
#  尺寸常量（基于 A4 正文宽 160mm）
# ─────────────────────────────────────────────────────────────────────────────

MM2IN  = 1.0 / 25.4
TW     = 160 * MM2IN          # 正文全宽 ≈ 6.30 in
TW_2_3 = TW * 2 / 3           # 三热图用：2/3 正文宽 ≈ 4.20 in
TW_1_2 = TW * 1 / 2           # 柱状图用：1/2 正文宽 ≈ 3.15 in

# ─────────────────────────────────────────────────────────────────────────────
#  全局字体样式（Times New Roman 10pt）
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'],
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    8,
    'figure.dpi':         600,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.7,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'lines.linewidth':    1.4,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

CASE_COLORS  = {'S1': '#d62728', 'S2': '#1f77b4', 'S4': '#2ca02c'}
CASE_MARKERS = {'S1': 'o',       'S2': 's',        'S4': '^'}

# ─────────────────────────────────────────────────────────────────────────────
#  数据层
# ─────────────────────────────────────────────────────────────────────────────

def load_voxler_csv(path):
    df = pd.read_csv(path, header=None, names=['x', 'y', 'z', 'v'])
    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    zs = np.sort(df['z'].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx * ny * nz != len(df):
        raise ValueError("网格不匹配")
    V = df.sort_values(['x', 'y', 'z'])['v'].values.reshape(nx, ny, nz)
    return V, xs, ys, zs


def build_slope_mask(xs, ys, zs):
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
    dx    = SLOPE_TOE_LENGTH - SLOPE_CREST_LENGTH
    x_max = SLOPE_TOE_LENGTH - (dx / SLOPE_HEIGHT) * Z3
    return (Z3 >= 0) & (Z3 <= SLOPE_HEIGHT) & (X3 >= 0) & (X3 <= x_max)


def largest_connected(V_masked, slope_mask, threshold, min_size=5):
    high       = (V_masked > threshold) & slope_mask
    labeled, _ = ndimage.label(high, structure=np.ones((3, 3, 3), dtype=int))
    if labeled.max() == 0:
        return 0
    sizes    = np.bincount(labeled.ravel())
    sizes[0] = 0
    sz = int(sizes.max())
    return sz if sz >= min_size else 0


def load_all_data():
    all_inside = []
    for path, *_ in CSV_FILES:
        if not os.path.exists(path):
            continue
        V, xs, ys, zs = load_voxler_csv(path)
        mask = build_slope_mask(xs, ys, zs)
        all_inside.append(V[mask])
    all_inside = np.concatenate(all_inside)
    global_thr = {p: float(np.percentile(all_inside, p))
                  for p in THRESHOLD_PERCENTILES}
    thr = global_thr[MAIN_PCT]

    per_case = {}
    for path, case, _, hours in CSV_FILES:
        if not os.path.exists(path):
            print(f'  [跳过] {path}')
            continue
        V, xs, ys, zs = load_voxler_csv(path)
        mask  = build_slope_mask(xs, ys, zs)
        V_m   = np.where(mask, V, np.nan)
        lmax  = largest_connected(V_m, mask, thr, min_size=MIN_COMP_VOXELS)
        per_case.setdefault(case, {'times': [], 'lmax': []})
        per_case[case]['times'].append(hours)
        per_case[case]['lmax'].append(lmax)

    for case in per_case:
        idx = np.argsort(per_case[case]['times'])
        per_case[case]['times'] = np.array(per_case[case]['times'])[idx]
        per_case[case]['lmax']  = np.array(per_case[case]['lmax'])[idx]
    return per_case

# ─────────────────────────────────────────────────────────────────────────────
#  拟合函数
# ─────────────────────────────────────────────────────────────────────────────

def logistic_func(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def exponential_func(t, a, lam):
    return a * np.exp(lam * t)

def linear_func(t, a, b):
    return a * t + b

def power_func(t, a, b):
    return a * np.power(np.maximum(t, 1e-9), b)


def aic_bic(y_true, y_pred, n_params):
    n    = len(y_true)
    sse  = np.sum((y_true - y_pred) ** 2)
    sse  = max(sse, 1e-12)
    k    = n_params
    aic  = n * np.log(sse / n) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    bic  = n * np.log(sse / n) + k * np.log(n)
    r2   = 1.0 - sse / max(np.sum((y_true - y_true.mean()) ** 2), 1e-12)
    return float(r2), float(aicc), float(bic)


def fit_logistic(times, lmax, window=None):
    t, y = _window_slice(times, lmax, window)
    if len(t) < 4:
        return None, np.nan, np.nan, np.nan
    K0   = float(y.max()) * 1.2
    t0_0 = float(t[len(t) // 2])
    try:
        popt, _ = optimize.curve_fit(
            logistic_func, t, y,
            p0=[K0, 1.0, t0_0],
            bounds=([1, 0.01, t.min() - 2], [y.max() * 10, 20, t.max() + 2]),
            maxfev=10000)
        y_fit = logistic_func(t, *popt)
        r2, aicc, bic = aic_bic(y, y_fit, 3)
        return popt, r2, aicc, bic
    except Exception:
        return None, np.nan, np.nan, np.nan


def fit_all_models(times, lmax, window=None):
    t, y = _window_slice(times, lmax, window)
    if len(t) < 3:
        return {}
    out = {}

    K0, t0_0 = float(y.max()) * 1.2, float(t[len(t) // 2])
    try:
        popt, _ = optimize.curve_fit(
            logistic_func, t, y,
            p0=[K0, 1.0, t0_0],
            bounds=([1, 0.01, t.min()-2], [y.max()*10, 20, t.max()+2]),
            maxfev=10000)
        yp = logistic_func(t, *popt)
        r2, aicc, bic = aic_bic(y, yp, 3)
        out['Logistic'] = dict(r2=r2, aicc=aicc, bic=bic)
    except Exception:
        out['Logistic'] = dict(r2=np.nan, aicc=np.nan, bic=np.nan)

    try:
        popt, _ = optimize.curve_fit(exponential_func, t, y,
                                     p0=[1.0, 0.5], maxfev=5000)
        yp = exponential_func(t, *popt)
        r2, aicc, bic = aic_bic(y, yp, 2)
        out['Exponential'] = dict(r2=r2, aicc=aicc, bic=bic)
    except Exception:
        out['Exponential'] = dict(r2=np.nan, aicc=np.nan, bic=np.nan)

    try:
        popt, _ = optimize.curve_fit(linear_func, t, y, maxfev=5000)
        yp = linear_func(t, *popt)
        r2, aicc, bic = aic_bic(y, yp, 2)
        out['Linear'] = dict(r2=r2, aicc=aicc, bic=bic)
    except Exception:
        out['Linear'] = dict(r2=np.nan, aicc=np.nan, bic=np.nan)

    t_pos = np.maximum(t - t.min() + 0.1, 0.1)
    try:
        popt, _ = optimize.curve_fit(power_func, t_pos, y,
                                     p0=[1.0, 1.0], maxfev=5000)
        yp = power_func(t_pos, *popt)
        r2, aicc, bic = aic_bic(y, yp, 2)
        out['Power-law'] = dict(r2=r2, aicc=aicc, bic=bic)
    except Exception:
        out['Power-law'] = dict(r2=np.nan, aicc=np.nan, bic=np.nan)

    return out


def _window_slice(times, lmax, window):
    t = np.array(times, dtype=float)
    y = np.array(lmax,  dtype=float)
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])
        t, y = t[m], y[m]
    return t, y


def bootstrap_logistic(times, lmax, window, n_boot=BOOTSTRAP_N,
                       seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    t, y = _window_slice(times, lmax, window)
    n    = len(t)
    if n < 4:
        return None
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tb, yb = t[idx], y[idx]
        try:
            K0   = float(yb.max()) * 1.2
            t0_0 = float(tb[len(tb) // 2])
            popt, _ = optimize.curve_fit(
                logistic_func, tb, yb,
                p0=[K0, 1.0, t0_0],
                bounds=([1, 0.01, tb.min()-2], [yb.max()*10, 20, tb.max()+2]),
                maxfev=5000)
            boot.append(popt)
        except Exception:
            continue
    if len(boot) < 20:
        return None
    boot = np.array(boot)
    ci   = np.percentile(boot, [2.5, 97.5], axis=0)
    med  = np.median(boot, axis=0)
    return {
        'K':  (med[0], ci[0, 0], ci[1, 0]),
        'r':  (med[1], ci[0, 1], ci[1, 1]),
        't0': (med[2], ci[0, 2], ci[1, 2]),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  图1：fig1_Lmax_logistic
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig1_lmax(per_case, save_prefix):
    cases = [c for c in ['S1', 'S2', 'S4'] if c in per_case]

    fig_w = TW
    fig_h = (TW / 3) * 1.25
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    for ax, case in zip(axes, cases):
        t_data  = per_case[case]['times']
        y_data  = per_case[case]['lmax']
        col     = CASE_COLORS[case]
        mkr     = CASE_MARKERS[case]
        t_fail  = FAILURE_HOURS.get(case)
        win     = FIT_WINDOWS.get(case)
        exp_win = EXP_WINDOWS.get(case)

        popt, r2, _, _ = fit_logistic(t_data, y_data, window=win)
        t_dense = np.linspace(min(t_data) * 0.95, max(t_data) * 1.01, 400)
        if popt is not None:
            K, r_val, t0 = popt
            ax.plot(t_dense, logistic_func(t_dense, *popt),
                    '-', color=col, lw=1.6, zorder=3,
                    label='Logistic fit')
            param_str = (f'$K$={K:.0f}\n'
                         f'$r$={r_val:.2f} h$^{{-1}}$\n'
                         f'$t_0$={t0:.1f} h\n'
                         f'$R^2$={r2:.3f}')
            ax.text(0.03, 0.60, param_str,
                    transform=ax.transAxes,
                    va='top', ha='left',
                    fontsize=7.5, linespacing=1.4,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.80,
                              edgecolor='#cccccc', linewidth=0.5))

        if exp_win is not None:
            t_arr, y_arr = np.array(t_data), np.array(y_data)
            em = (t_arr >= exp_win[0]) & (t_arr <= exp_win[1])
            te, ye = t_arr[em], y_arr[em]
            lam_label = EXP_LAMBDAS.get(case)
            if lam_label is not None and len(te) >= 2 and ye.min() > 0:
                try:
                    coeffs = np.polyfit(te, np.log(np.maximum(ye, 1)), 1)
                    t_ep   = np.linspace(exp_win[0], exp_win[1], 100)
                    ax.plot(t_ep, np.exp(np.polyval(coeffs, t_ep)),
                            '--', color='#444444', lw=1.1, zorder=4,
                            label=f'Exp. approx.\n'
                                  f'($\\lambda$={lam_label:.3f} h$^{{-1}}$)')
                except Exception:
                    pass

        ax.scatter(t_data, y_data,
                   color=col, marker=mkr, s=28, zorder=5,
                   edgecolors='k', linewidths=0.4,
                   label='Observed $L_{\\mathrm{max}}$')

        if t_fail is not None:
            ax.axvline(t_fail, color='k', ls='-', lw=0.9, zorder=4)
            ylim_top = ax.get_ylim()[1]
            ax.text(t_fail - 0.08, ylim_top * 0.97,
                    '$t_{\\mathrm{vis}}$',
                    va='top', ha='right', fontsize=8)

        ax.set_xlabel('Time since rain onset (h)', labelpad=2)
        ax.set_ylabel('$L_{\\mathrm{max}}$ (voxels)', labelpad=2)
        ax.set_title(f'Case {case}', pad=3)
        ax.grid(True, lw=0.35, alpha=0.45)
        ax.legend(loc='upper left', framealpha=0.85,
                  handlelength=1.4, borderpad=0.5,
                  labelspacing=0.3, fontsize=7.5)
        ax.tick_params(axis='both', which='major', pad=2)

    fig.tight_layout(pad=0.4, w_pad=0.9)
    _save(fig, save_prefix, 'fig1_Lmax_logistic')


# ─────────────────────────────────────────────────────────────────────────────
#  图2：fig2_logistic_params
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig2_params(per_case, save_prefix):
    cases = [c for c in ['S1', 'S2', 'S4'] if c in per_case]

    print('\n[Bootstrap] 计算 Logistic 参数 CI...')
    ci_res = {}
    for case in cases:
        print(f'  {case}')
        ci_res[case] = bootstrap_logistic(
            per_case[case]['times'],
            per_case[case]['lmax'],
            window=FIT_WINDOWS.get(case),
            n_boot=BOOTSTRAP_N)
        if ci_res[case]:
            for pk in ('K', 'r', 't0'):
                med, lo, hi = ci_res[case][pk]
                print(f'    {pk}: {med:.3f}  [{lo:.3f}, {hi:.3f}]')

    param_keys    = ['K',  'r',  't0']
    param_ylabels = [
        '$K$ (saturation capacity, voxels)',
        '$r$ (growth rate, h$^{-1}$)',
        '$t_0$ (inflection time, h)',
    ]

    fig_w = TW
    fig_h = TW / 3 * 1.35
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    x  = np.arange(len(cases))
    bw = 0.45

    for ax, pk, yl in zip(axes, param_keys, param_ylabels):
        meds, lo_e, hi_e = [], [], []
        for case in cases:
            ci = ci_res.get(case)
            if ci and pk in ci:
                med, lo, hi = ci[pk]
                meds.append(med)
                lo_e.append(med - lo)
                hi_e.append(hi  - med)
            else:
                meds.append(np.nan); lo_e.append(0); hi_e.append(0)

        ax.bar(
            x, meds,
            yerr=[lo_e, hi_e],
            color=[CASE_COLORS[c] for c in cases],
            error_kw=dict(elinewidth=1.0, capsize=4, capthick=1.0, ecolor='k'),
            edgecolor='k', linewidth=0.5, width=bw, zorder=3)

        for xi, (med, he) in enumerate(zip(meds, hi_e)):
            if not np.isnan(med):
                if abs(med) >= 1000:
                    fmt = f'{med:.0f}'
                elif abs(med) >= 10:
                    fmt = f'{med:.1f}'
                else:
                    fmt = f'{med:.2f}'
                ax.text(xi, med + he * 1.05 + abs(med) * 0.02,
                        fmt, ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.set_xlabel('Case', labelpad=2)
        ax.set_ylabel(yl, labelpad=2)
        ax.grid(axis='y', lw=0.35, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', pad=2)
        ax.set_ylim(bottom=0)

    for ax in axes:
        ax.set_title('')

    fig.suptitle('Logistic model parameters with 95% bootstrap CI',
                 fontsize=10, y=1.01)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    _save(fig, save_prefix, 'fig2_logistic_params')


# ─────────────────────────────────────────────────────────────────────────────
#  计算 fig3 所需指标矩阵（供 fig3a / fig3b 共用）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(per_case):
    cases  = [c for c in ['S1', 'S2', 'S4'] if c in per_case]
    models = ['Logistic', 'Exponential', 'Linear', 'Power-law']

    metrics = {case: {} for case in cases}
    for case in cases:
        res = fit_all_models(per_case[case]['times'],
                             per_case[case]['lmax'],
                             window=FIT_WINDOWS.get(case))
        for m in models:
            metrics[case][m] = res.get(
                m, dict(r2=np.nan, aicc=np.nan, bic=np.nan))

    r2_mat   = np.array([[metrics[c][m]['r2']   for m in models] for c in cases])
    aicc_mat = np.array([[metrics[c][m]['aicc'] for m in models] for c in cases])
    bic_mat  = np.array([[metrics[c][m]['bic']  for m in models] for c in cases])
    return cases, models, r2_mat, aicc_mat, bic_mat


# ─────────────────────────────────────────────────────────────────────────────
#  图3a：fig3a_model_heatmaps
#  三热图并排，全幅宽（160mm），行高充裕，x 轴标签不被截断
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3a_heatmaps(per_case, save_prefix):
    cases, models, r2_mat, aicc_mat, bic_mat = _compute_metrics(per_case)
    nrow  = len(cases)
    mlbls = models

    # 宽=全幅，高按行数自适应（每行 0.72 in + 上下边距 1.1 in）
    fig_w = TW
    fig_h = nrow * 0.72 + 1.1
    fig_h = max(fig_h, 2.8)

    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.35))       # 热图间距，为 colorbar 留空间

    def draw_hm(ax, mat, title, cmap, vmin, vmax, fmt, show_y):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(mlbls, rotation=40, ha='right',
                           fontsize=8.5, rotation_mode='anchor')
        ax.set_yticks(range(nrow))
        ax.set_yticklabels(cases if show_y else [''] * nrow, fontsize=9)
        ax.set_title(title, fontsize=9, pad=4)

        # 单元格数值
        for i in range(nrow):
            for j in range(len(models)):
                v   = mat[i, j]
                txt = 'N/A' if np.isnan(v) else format(v, fmt)
                nv  = (v - vmin) / max(vmax - vmin, 1e-9)
                fc  = 'white' if nv > 0.55 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=fc, fontweight='bold')

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7.5)

    draw_hm(axes[0], r2_mat,
            '$R^2$ (higher = better)', 'RdYlGn',
            0.0, 1.0, '.3f', show_y=True)
    draw_hm(axes[1], aicc_mat,
            'AICc (lower = better)', 'RdYlGn_r',
            np.nanmin(aicc_mat) - 3, np.nanmax(aicc_mat) + 3,
            '.1f', show_y=False)
    draw_hm(axes[2], bic_mat,
            'BIC (lower = better)', 'RdYlGn_r',
            np.nanmin(bic_mat) - 3, np.nanmax(bic_mat) + 3,
            '.1f', show_y=False)

    fig.suptitle('Goodness-of-fit: Logistic vs. alternative models',
                 fontsize=10, y=1.01)

    # bottom 留足空间给旋转的 x 轴标签
    fig.tight_layout(pad=0.5, w_pad=1.2,
                     rect=[0, 0.05, 1, 1])
    _save(fig, save_prefix, 'fig3a_model_heatmaps')


# ─────────────────────────────────────────────────────────────────────────────
#  图3b：fig3b_delta_aicc
#  ΔAICc 柱状图，半幅宽（80mm），高适中
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig3b_delta_aicc(per_case, save_prefix):
    cases, models, r2_mat, aicc_mat, bic_mat = _compute_metrics(per_case)

    li         = models.index('Logistic')
    delta_aicc = aicc_mat - aicc_mat[:, li:li+1]
    other_m    = [m for m in models if m != 'Logistic']
    other_i    = [i for i, m in enumerate(models) if m != 'Logistic']
    delta_oth  = delta_aicc[:, other_i]

    # 半幅宽，高比宽略小（约黄金比）
    fig_w = TW_1_2
    fig_h = fig_w * 0.88
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x    = np.arange(len(cases))
    bw   = 0.22
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c']
    offs = np.array([-1, 0, 1]) * bw

    for oi, (om, off, bc) in enumerate(zip(other_m, offs, cols)):
        vals = delta_oth[:, oi]
        ax.bar(x + off, vals, width=bw, color=bc, alpha=0.85,
               label=om, edgecolor='k', linewidth=0.4, zorder=3)

    ax.axhline(0, color='k', lw=0.8, ls='--', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=9)
    ax.set_xlabel('Case', labelpad=2)
    ax.set_ylabel('$\\Delta$AICc (model $-$ Logistic)', labelpad=2)
    ax.set_title('$\\Delta$AICc vs. Logistic', fontsize=10, pad=4)
    ax.text(0.98, 0.97,
            'positive = worse than Logistic',
            transform=ax.transAxes,
            va='top', ha='right', fontsize=7.5,
            color='#555555')
    ax.legend(fontsize=8, loc='upper left',
              framealpha=0.85, handlelength=1.2,
              borderpad=0.5, labelspacing=0.3)
    ax.grid(axis='y', lw=0.35, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', pad=2)

    fig.tight_layout(pad=0.5)
    _save(fig, save_prefix, 'fig3b_delta_aicc')


# ─────────────────────────────────────────────────────────────────────────────
#  保存辅助
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, prefix, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(prefix, f'{name}.{ext}')
        fig.savefig(p)
        print(f'  已保存: {p}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(' 加载数据...')
    per_case = load_all_data()
    if not per_case:
        print('[错误] 未能加载任何数据，请检查 CSV_FILES 路径。')
        return
    for case, d in per_case.items():
        print(f'  {case}: {len(d["times"])} 个时刻  '
              f'Lmax ∈ [{d["lmax"].min()}, {d["lmax"].max()}]')

    print('\n 图1：Lmax Logistic 拟合...')
    plot_fig1_lmax(per_case, OUTPUT_DIR)

    print('\n 图2：Logistic 参数 Bootstrap CI...')
    plot_fig2_params(per_case, OUTPUT_DIR)

    print('\n 图3a：三热图...')
    plot_fig3a_heatmaps(per_case, OUTPUT_DIR)

    print('\n 图3b：ΔAICc 柱状图...')
    plot_fig3b_delta_aicc(per_case, OUTPUT_DIR)

    print(f'\n完成。输出目录: {OUTPUT_DIR}')
    print('  fig1_Lmax_logistic.pdf/.png      → 替换原图5(b)')
    print('  fig2_logistic_params.pdf/.png    → 替换原图5(c)')
    print('  fig3a_model_heatmaps.pdf/.png    → 三热图（独立）')
    print('  fig3b_delta_aicc.pdf/.png        → ΔAICc柱状图（独立）')


if __name__ == '__main__':
    main()
