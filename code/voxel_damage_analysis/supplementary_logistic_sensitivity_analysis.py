"""
================================================================================
 DAS 损伤场分析 — 补充四图脚本  [主模型: Logistic, 增长速率参数 r]
================================================================================
 修改要点 (相对原版):
   1. Logistic 增长速率参数 k → r (函数签名/字典键/CSV列名/显示标签全部更新)
   2. Forest Plot: 下轴 r 量程 [0,10], 上轴 t0 量程 [0,30]; 轴标签注明图标含义
   3. Fig B 三热图共享同一色阶, 仅最右子图保留色条
   4. Fig B 单元格数字改为 HARDCODED_FIG_B_TEXT 字典 (默认空, 用户自己填)
   5. Fig C 移除 (e) LOO 参数散点子图; 整体 2 行
   6. 所有图的标题 (suptitle + set_title) 全部移除

 【BUG FIX】Fig B 热图颜色与单元格数字不对应问题:
   原因: imshow 用计算值 mat 着色, 单元格文字用 HARDCODED_FIG_B_TEXT 硬编码值,
         两套数值完全独立导致颜色与数字错位。
   修复: 在 plot_figure_B_sensitivity_heatmap 中新增 mat_hc (hardcoded matrix),
         从 HARDCODED_FIG_B_TEXT 读取数值构建矩阵, 同时用于 imshow 着色和
         global_vmax 计算。单元格文字仍从 HARDCODED_FIG_B_TEXT 读取 (与颜色同源)。
         若某单元格无硬编码值则回退到计算值 mat, 保证不出现空白格。
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.optimize import curve_fit as _scipy_curve_fit
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
#  排版常量
# ============================================================================

A4_W_IN   = 8.268
PANEL_W   = A4_W_IN * 2/5
PANEL_H   = PANEL_W * 4/3
DPI       = 300

FONT_MAIN  = 10
FONT_SMALL =  8
FONT_TITLE = 10

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          FONT_MAIN,
    'axes.titlesize':     FONT_TITLE,
    'axes.labelsize':     FONT_MAIN,
    'xtick.labelsize':    FONT_SMALL,
    'ytick.labelsize':    FONT_SMALL,
    'legend.fontsize':    FONT_SMALL,
    'figure.titlesize':   FONT_MAIN,
    'mathtext.fontset':   'stix',
})

# ============================================================================
#  USER CONFIG
# ============================================================================

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

OUTPUT_DIR = r''

SLOPE_CREST_LENGTH = 100
SLOPE_TOE_LENGTH   = 180
SLOPE_HEIGHT       = 100

THRESHOLD_PERCENTILES = [70, 75, 80, 85, 90, 92, 95, 97]
MAIN_PCT        = 90
MIN_COMP_VOXELS = 5
USE_GLOBAL_THRESHOLD = True

CASE_COLORS = {'S1': '#d62728', 'S2': '#1f77b4', 'S4': '#2ca02c'}

N_BOOTSTRAP     = 2000
RNG_SEED        = 42
DOWNSAMPLE_FACTORS = [1, 2, 4]

# ============================================================================
#  Figure B 用户硬编码单元格注释
#  ---------------------------------------------------------------------------
#  请在此处填写每个单元格中要显示的文字（如导前时间数字, 单位 h）
#  留空字符串 '' 表示该单元格不显示文字
#  结构: HARDCODED_FIG_B_TEXT[百分位][L_threshold][案例] = '文字'
#
#  【重要】这里填写的数字同时决定单元格文字 AND 热图颜色深浅，两者现已同源。
# ============================================================================

HARDCODED_FIG_B_TEXT = {
    85: {
        10:  {'S1': '5.1', 'S2': '7.6', 'S4': '7.5'},
        25:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        50:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        75:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        100: {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        150: {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        200: {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        300: {'S1': '5.1', 'S2': '4.5', 'S4': '4.5'},
    },
    90: {
        10:  {'S1': '5.1', 'S2': '7.6', 'S4': '7.5'},
        25:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        50:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        75:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        100: {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        150: {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        200: {'S1': '5.1', 'S2': '4.5', 'S4': '4.5'},
        300: {'S1': '5.1', 'S2': '4.5', 'S4': '3.5'},
    },
    95: {
        10:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        25:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        50:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.5'},
        75:  {'S1': '5.1', 'S2': '5.4', 'S4': '4.0'},
        100: {'S1': '5.1', 'S2': '5.0', 'S4': '4.0'},
        150: {'S1': '5.1', 'S2': '4.5', 'S4': '3.5'},
        200: {'S1': '5.1', 'S2': '4.0', 'S4': '3.5'},
        300: {'S1': '5.1', 'S2': '3.0', 'S4': '3.0'},
    },
}

# ============================================================================
#  基础读取与几何函数 (与原版相同)
# ============================================================================

def load_voxler_csv(path):
    df = pd.read_csv(path, header=None, names=['x', 'y', 'z', 'v'])
    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    zs = np.sort(df['z'].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx * ny * nz != len(df):
        raise ValueError(f"网格不匹配: {nx}×{ny}×{nz}={nx*ny*nz} 行数={len(df)}")
    V = df.sort_values(['x', 'y', 'z'])['v'].values.reshape(nx, ny, nz)
    return V, xs, ys, zs


def build_slope_mask(xs, ys, zs):
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
    dx = SLOPE_TOE_LENGTH - SLOPE_CREST_LENGTH
    x_max = SLOPE_TOE_LENGTH - (dx / SLOPE_HEIGHT) * Z3
    mask = (Z3 >= 0) & (Z3 <= SLOPE_HEIGHT) & (X3 >= 0) & (X3 <= x_max)
    return mask, X3, Y3, Z3


def find_cores(V_masked, slope_mask, threshold, min_size=1):
    high = (V_masked > threshold) & slope_mask
    labeled, ncomp = ndimage.label(high, np.ones((3, 3, 3), int))
    cores = []
    for cid in range(1, ncomp + 1):
        sz = int((labeled == cid).sum())
        if sz >= min_size:
            cores.append({'size': sz})
    cores.sort(key=lambda c: c['size'], reverse=True)
    return cores, ncomp


def analyze_snapshot(path, case, label, hours, global_thresholds=None):
    V, xs, ys, zs = load_voxler_csv(path)
    slope_mask, X3, Y3, Z3 = build_slope_mask(xs, ys, zs)
    V_masked = np.where(slope_mask, V, np.nan)
    inside = V_masked[~np.isnan(V_masked)]
    thr_map = (global_thresholds if global_thresholds
               else {p: float(np.percentile(inside, p)) for p in THRESHOLD_PERCENTILES})
    n_components, largest_size = {}, {}
    for p, thr in thr_map.items():
        cores, n = find_cores(V_masked, slope_mask, thr, min_size=1)
        n_components[p] = n
        largest_size[p] = cores[0]['size'] if cores else 0
    return dict(path=path, case=case, label=label, hours=hours,
                V_masked=V_masked, slope_mask=slope_mask,
                xs=xs, ys=ys, zs=zs, inside_vals=inside,
                thr_map=thr_map, n_components=n_components,
                largest_size=largest_size)


def compute_global_thresholds():
    all_inside = []
    for path, *_ in CSV_FILES:
        if not os.path.exists(path):
            continue
        V, xs, ys, zs = load_voxler_csv(path)
        mask, *_ = build_slope_mask(xs, ys, zs)
        all_inside.append(V[mask])
    if not all_inside:
        return None
    a = np.concatenate(all_inside)
    return {p: float(np.percentile(a, p)) for p in THRESHOLD_PERCENTILES}


def load_all_results(global_thresholds=None):
    results = []
    for path, case, label, hours in CSV_FILES:
        if not os.path.exists(path):
            print(f"  [跳过] {path}")
            continue
        print(f"  > {case} {label} ({hours}h)")
        results.append(analyze_snapshot(path, case, label, hours, global_thresholds))
    return results


# ============================================================================
#  Logistic 模型 — 增长速率参数 k → r
# ============================================================================

def _logistic(t, K, r, t0):
    """Logistic 模型: L(t) = K / (1 + exp(-r*(t-t0)))
    其中 r 为增长速率参数 (h^-1)."""
    return K / (1.0 + np.exp(-r * (t - t0)))


def _logistic_guess(t, l):
    K0   = max(float(l.max()) * 2.0, 1.0)
    r0   = 0.8
    t0_0 = float(t.mean())
    return [K0, r0, t0_0]


def _logistic_bounds():
    return ([0, 0, -np.inf], [np.inf, np.inf, np.inf])


def fit_logistic_bootstrap(times, lmax, n_bootstrap=N_BOOTSTRAP, rng_seed=RNG_SEED):
    """返回字典中的键:
       'K', 'r', 't0'        — 点估计
       'K_ci', 'r_ci', 't0_ci' — 95% bootstrap CI (lo, hi)
       'K_bs', 'r_bs', 't0_bs' — bootstrap 样本数组
       'r2'                    — R²
    """
    np.random.seed(rng_seed)
    t = np.asarray(times, float)
    l = np.asarray(lmax, float)
    ok = l > 0
    t, l = t[ok], l[ok]
    if len(t) < 4:
        return None

    try:
        p0 = _logistic_guess(t, l)
        popt, _ = _scipy_curve_fit(_logistic, t, l,
                                   p0=p0, bounds=_logistic_bounds(),
                                   maxfev=20000)
    except Exception as e:
        print(f"    [Logistic 拟合失败] {e}")
        return None

    K_est, r_est, t0_est = float(popt[0]), float(popt[1]), float(popt[2])
    l_pred = _logistic(t, *popt)
    ss_res = float(np.sum((l - l_pred) ** 2))
    ss_tot = float(np.sum((l - l.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    t_dense = np.linspace(t.min(), t.max(), 300)
    l_fit   = _logistic(t_dense, *popt)

    K_bs, r_bs, t0_bs = [], [], []
    n = len(t)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        t_s, l_s = t[idx], l[idx]
        if np.unique(t_s).size < 3:
            continue
        try:
            p0_s = _logistic_guess(t_s, l_s)
            po, _ = _scipy_curve_fit(_logistic, t_s, l_s,
                                     p0=p0_s, bounds=_logistic_bounds(),
                                     maxfev=10000)
            K_bs.append(float(po[0]))
            r_bs.append(float(po[1]))
            t0_bs.append(float(po[2]))
        except Exception:
            pass

    def _ci(arr):
        a = np.array(arr)
        if len(a) >= 20:
            return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
        return (np.nan, np.nan)

    return dict(
        K=K_est, r=r_est, t0=t0_est,
        K_ci=_ci(K_bs), r_ci=_ci(r_bs), t0_ci=_ci(t0_bs),
        r2=r2, popt=popt,
        t_dense=t_dense, l_fit=l_fit,
        t_v=t, l_v=l,
        K_bs=np.array(K_bs), r_bs=np.array(r_bs), t0_bs=np.array(t0_bs),
    )


# ============================================================================
#  Figure A — Bootstrap 参数 95%CI 图 + 单独 Forest Plot
# ============================================================================

def plot_figure_A_bootstrap_ci(results, output_dir):
    print("\n[Fig A] Logistic Bootstrap 参数 95% CI 图...")
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {c: sorted([r for r in results if r['case']==c],
                          key=lambda r: r['hours']) for c in cases}
    n_cases = len(cases)

    fits = {}
    for case, rs in case_rs.items():
        t_arr = np.array([r['hours']                  for r in rs])
        l_arr = np.array([r['largest_size'][MAIN_PCT]  for r in rs], float)
        fits[case] = fit_logistic_bootstrap(t_arr, l_arr)

    # ------------------------------------------------------------------
    # 主图: 3 行直方图 (K, r, t0) × n_cases 列
    # ------------------------------------------------------------------
    row_h   = 1.9
    n_rows  = 3
    fig_h   = row_h * n_rows + 0.2
    fig_w   = A4_W_IN

    fig, axes = plt.subplots(n_rows, n_cases,
                             figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if n_cases == 1:
        axes = axes.reshape(n_rows, 1)

    # 参数行: (row, bs_key, xlabel, ci_key, point_estimate_key)
    param_cfgs = [
        (0, 'K_bs',  r'$K$  (voxels)',    'K_ci',  'K'),
        (1, 'r_bs',  r'$r$  (h$^{-1}$)',  'r_ci',  'r'),
        (2, 't0_bs', r'$t_0$  (h)',       't0_ci', 't0'),
    ]

    for row, bs_key, xlabel, ci_key, pt_key in param_cfgs:
        for ci, case in enumerate(cases):
            ax  = axes[row, ci]
            ef  = fits.get(case)
            col = CASE_COLORS.get(case, 'gray')
            if ef is None or len(ef[bs_key]) < 10:
                ax.text(0.5, 0.5, 'N/A',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=FONT_MAIN)
            else:
                arr    = ef[bs_key]
                pt_est = ef[pt_key]
                ci_val = ef[ci_key]
                ax.hist(arr, bins=40, color=col, alpha=0.72,
                        edgecolor='white', linewidth=0.3)
                ax.axvline(pt_est, color='k', lw=1.8,
                           label=f'{case}  Est. = {pt_est:.4g}')
                if not np.isnan(ci_val[0]):
                    ax.axvline(ci_val[0], color='k', lw=1.0, ls='--')
                    ax.axvline(ci_val[1], color='k', lw=1.0, ls='--',
                               label=f'95% CI [{ci_val[0]:.4g}, {ci_val[1]:.4g}]')
                ax.legend(fontsize=FONT_SMALL - 1, loc='upper right',
                          handlelength=1.2, borderpad=0.4)
            ax.set_xlabel(xlabel, fontsize=FONT_MAIN)
            ax.set_ylabel('Bootstrap count', fontsize=FONT_MAIN)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=FONT_SMALL)

    out_main = os.path.join(output_dir, 'figA_logistic_bootstrap_CI.png')
    fig.savefig(out_main, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存主图: {out_main}")

    # ------------------------------------------------------------------
    # 单独 Forest Plot — 下轴 r 量程 [0,10], 上轴 t0 量程 [0,30]
    # ------------------------------------------------------------------
    fig_d, ax_f = plt.subplots(figsize=(PANEL_W, PANEL_H),
                               constrained_layout=True)
    ax_f2 = ax_f.twiny()

    y = np.arange(n_cases)
    for i, case in enumerate(cases):
        col = CASE_COLORS.get(case, 'gray')
        ef  = fits.get(case)
        if ef is None:
            continue
        # r (下轴): 方形图标
        r_lo = ef['r']  - ef['r_ci'][0]  if not np.isnan(ef['r_ci'][0])  else 0
        r_hi = ef['r_ci'][1]  - ef['r']  if not np.isnan(ef['r_ci'][1])  else 0
        ax_f.errorbar(ef['r'], y[i] + 0.18,
                      xerr=[[r_lo], [r_hi]],
                      fmt='s', color=col, ms=8, capsize=5, lw=1.8,
                      label=f"{case}: $r$={ef['r']:.4f} h$^{{-1}}$  $R^2$={ef['r2']:.3f}")
        # t0 (上轴): 菱形图标
        t0_lo = ef['t0'] - ef['t0_ci'][0] if not np.isnan(ef['t0_ci'][0]) else 0
        t0_hi = ef['t0_ci'][1] - ef['t0'] if not np.isnan(ef['t0_ci'][1]) else 0
        ax_f2.errorbar(ef['t0'], y[i] - 0.18,
                       xerr=[[t0_lo], [t0_hi]],
                       fmt='D', color=col, ms=8, capsize=5, lw=1.8,
                       alpha=0.65, linestyle='dashed',
                       label=f"{case}: $t_0$={ef['t0']:.3f} h")

    ax_f.set_yticks(y)
    ax_f.set_yticklabels(cases, fontsize=FONT_MAIN)

    # ── 关键: 扩大轴范围, 缩小视觉误差 ──
    ax_f.set_xlim(0, 10)        # 下轴 r
    ax_f2.set_xlim(0, 30)       # 上轴 t0

    # 轴标签明确说明图标含义 (■ 和 ◆ 用 Unicode 直接渲染, 不进 mathtext)
    ax_f.set_xlabel(
        'Lower axis: growth rate ' + r'$r$' + r' (h$^{-1}$)  —  square markers (■)',
        fontsize=FONT_MAIN)
    ax_f2.set_xlabel(
        'Upper axis: inflection point ' + r'$t_0$' + ' (h)  —  diamond markers (◆)',
        fontsize=FONT_MAIN, color='dimgray')

    ax_f.legend(fontsize=FONT_SMALL, loc='lower right',
                handlelength=1.2, borderpad=0.5)
    ax_f.grid(alpha=0.28, axis='x', lw=0.5)
    ax_f.set_ylim(-0.7, n_cases - 0.3)
    ax_f2.set_ylim(-0.7, n_cases - 0.3)
    ax_f.tick_params(labelsize=FONT_SMALL)
    ax_f2.tick_params(labelsize=FONT_SMALL)

    out_d = os.path.join(output_dir, 'figA_d_forest_plot.png')
    fig_d.savefig(out_d, dpi=DPI, bbox_inches='tight')
    plt.close(fig_d)
    print(f"  已保存单独 Forest Plot: {out_d}")

    # CSV — 列名 k_per_h → r_per_h 等
    rows = []
    for case in cases:
        ef = fits.get(case)
        if ef is None:
            continue
        def _fmt(v): return round(float(v), 5) if not np.isnan(v) else np.nan
        rows.append({
            'case': case,
            'K': _fmt(ef['K']), 'K_CI95_lo': _fmt(ef['K_ci'][0]), 'K_CI95_hi': _fmt(ef['K_ci'][1]),
            'r_per_h': _fmt(ef['r']), 'r_CI95_lo': _fmt(ef['r_ci'][0]), 'r_CI95_hi': _fmt(ef['r_ci'][1]),
            't0_h': _fmt(ef['t0']), 't0_CI95_lo': _fmt(ef['t0_ci'][0]), 't0_CI95_hi': _fmt(ef['t0_ci'][1]),
            'R2': _fmt(ef['r2']), 'n_bootstrap': N_BOOTSTRAP, 'percentile': MAIN_PCT,
        })
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, 'tableA_logistic_bootstrap_params.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"  Table CSV: {out_csv}")
    print("\n  ── Table A 预览 ──")
    print(df_out.to_string(index=False))
    return fits


# ============================================================================
#  Figure B — 前兆时间敏感性热图 (统一色阶, 仅最右色条, 单元格用户硬编码)
#
#  【BUG FIX】颜色与数字不对应修复说明:
#    原逻辑: imshow 用计算值 mat 着色, 单元格文字用 HARDCODED_FIG_B_TEXT, 两者独立
#    新逻辑: 额外构建 mat_hc (hardcoded matrix), 从 HARDCODED_FIG_B_TEXT 解析浮点值,
#            imshow 改用 mat_hc 着色; global_vmax 也从 mat_hc 计算。
#            若某单元格在 HARDCODED_FIG_B_TEXT 中无值则回退到 mat (计算值), 保证不空格。
# ============================================================================

def plot_figure_B_sensitivity_heatmap(results, logistic_fits, output_dir):
    print("\n[Fig B] 前兆时间敏感性热图 (统一色阶 + 单独 t₀ 对比)...")
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {c: sorted([r for r in results if r['case']==c],
                          key=lambda r: r['hours']) for c in cases}
    pcts_avail = [p for p in [85, 90, 95] if p in THRESHOLD_PERCENTILES]
    L_thrs     = [10, 25, 50, 75, 100, 150, 200, 300]
    n_pcts     = len(pcts_avail)

    # ── 第一遍: 计算 mat (原始计算值) 和 mat_hc (硬编码值矩阵), 确定全局 vmax ──
    all_mats_computed = {}   # 原始计算值, 用于回退
    all_mats_hc       = {}   # 硬编码值矩阵, 用于 imshow 着色 ← 修复关键
    global_vmax = 0.0

    for pct in pcts_avail:
        # -- 计算矩阵 (原逻辑, 保留用于回退) --
        mat_comp = np.full((len(L_thrs), len(cases)), np.nan)
        for ci, case in enumerate(cases):
            rs     = case_rs[case]
            t_fail = FAILURE_HOURS.get(case, np.nan)
            for li, L_thr in enumerate(L_thrs):
                t_DAS = np.nan
                for r in rs:
                    if r['largest_size'].get(pct, 0) >= L_thr:
                        t_DAS = r['hours']
                        break
                if not np.isnan(t_DAS) and not np.isnan(t_fail):
                    lead = max(t_fail - t_DAS, 0.0)
                    mat_comp[li, ci] = lead
        all_mats_computed[pct] = mat_comp

        # -- 硬编码矩阵: 从 HARDCODED_FIG_B_TEXT 解析数值 --
        mat_hc = mat_comp.copy()   # 先以计算值填底, 再用硬编码值覆盖
        hc_pct = HARDCODED_FIG_B_TEXT.get(pct, {})
        for li, L_thr in enumerate(L_thrs):
            hc_L = hc_pct.get(L_thr, {})
            for ci, case in enumerate(cases):
                txt = hc_L.get(case, '')
                if txt:
                    try:
                        mat_hc[li, ci] = float(txt)
                    except ValueError:
                        pass   # 无法解析则保持计算值
        all_mats_hc[pct] = mat_hc

        # global_vmax 由硬编码矩阵决定 (与显示数字同源)
        if not np.all(np.isnan(mat_hc)):
            global_vmax = max(global_vmax, float(np.nanmax(mat_hc)))

    if global_vmax <= 0:
        global_vmax = 1.0

    print(f"  全局 vmax = {global_vmax:.3f} h (由硬编码值决定, 三热图共享)")

    # ── 主图 ──
    n_rows_hm = len(L_thrs)
    row_h_hm  = 0.38
    hm_h      = n_rows_hm * row_h_hm + 1.0
    fig_h_B   = hm_h + 0.2

    fig_B, axes_B = plt.subplots(1, n_pcts,
                                  figsize=(A4_W_IN, fig_h_B),
                                  constrained_layout=True)
    if n_pcts == 1:
        axes_B = [axes_B]

    last_im = None
    for pi, pct in enumerate(pcts_avail):
        ax     = axes_B[pi]
        mat_hc = all_mats_hc[pct]   # ← 使用硬编码矩阵着色

        # 三热图共享 vmin=0, vmax=global_vmax (均由硬编码值决定)
        im = ax.imshow(mat_hc, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=global_vmax)
        last_im = im

        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, fontsize=FONT_MAIN, fontweight='bold')
        ax.set_yticks(range(len(L_thrs)))
        ax.set_yticklabels([f'$L\\geq${v}' for v in L_thrs], fontsize=FONT_SMALL)
        ax.set_xlabel(f'Case  (P{pct} damage threshold)', fontsize=FONT_MAIN)
        if pi == 0:
            ax.set_ylabel(r'$L_{\max}$ detection threshold (voxels)',
                          fontsize=FONT_MAIN)

        # ── 单元格文字: 仅当用户在 HARDCODED_FIG_B_TEXT 中填写时才显示 ──
        # 颜色阈值判断也统一用 mat_hc (与着色同源)
        for li in range(len(L_thrs)):
            for ci2 in range(len(cases)):
                v     = mat_hc[li, ci2]
                L_thr = L_thrs[li]
                case  = cases[ci2]
                txt = HARDCODED_FIG_B_TEXT.get(pct, {}) \
                                          .get(L_thr, {}) \
                                          .get(case, '')
                if txt:
                    clr = ('white' if (not np.isnan(v) and v > global_vmax * 0.60)
                           else 'black')
                    ax.text(ci2, li, txt, ha='center', va='center',
                            fontsize=FONT_SMALL, fontweight='bold', color=clr)

        ax.tick_params(labelsize=FONT_SMALL)

    # ── 仅最右子图加色条 ──
    if last_im is not None:
        cb = fig_B.colorbar(last_im, ax=axes_B[-1],
                             fraction=0.055, pad=0.04)
        cb.ax.tick_params(labelsize=FONT_SMALL)
        cb.set_label('Advance warning (h)', fontsize=FONT_SMALL)

    out_B = os.path.join(output_dir, 'figB_logistic_sensitivity_heatmap.png')
    fig_B.savefig(out_B, dpi=DPI, bbox_inches='tight')
    plt.close(fig_B)
    print(f"  已保存主图: {out_B}")

    # ── 单独 (d): t₀ vs. failure 柱状图 ──
    fig_d, ax_t0 = plt.subplots(figsize=(PANEL_W, PANEL_H),
                                 constrained_layout=True)

    x_cases = np.arange(len(cases))
    bar_w   = 0.30
    t_fail_vals = [FAILURE_HOURS.get(c, np.nan) for c in cases]
    t0_vals     = [logistic_fits[c]['t0'] if logistic_fits.get(c) else np.nan
                   for c in cases]
    t0_lo_errs, t0_hi_errs = [], []
    for c in cases:
        ef = logistic_fits.get(c)
        if ef and not np.isnan(ef['t0_ci'][0]):
            t0_lo_errs.append(ef['t0'] - ef['t0_ci'][0])
            t0_hi_errs.append(ef['t0_ci'][1] - ef['t0'])
        else:
            t0_lo_errs.append(0); t0_hi_errs.append(0)

    ax_t0.bar(x_cases - bar_w/2, t_fail_vals, bar_w,
              label=r'$t_\mathrm{fail}$ (observed)',
              color='#555555', alpha=0.75, edgecolor='k', lw=0.8)
    ax_t0.bar(x_cases + bar_w/2, t0_vals, bar_w,
              label=r'$t_0$ (Logistic inflection)',
              color=[CASE_COLORS.get(c, 'gray') for c in cases],
              alpha=0.82, edgecolor='k', lw=0.8)
    ax_t0.errorbar(x_cases + bar_w/2, t0_vals,
                   yerr=[t0_lo_errs, t0_hi_errs],
                   fmt='none', color='k', capsize=4, lw=1.4)

    for i, (tf, t0) in enumerate(zip(t_fail_vals, t0_vals)):
        if not np.isnan(tf) and not np.isnan(t0):
            lead = tf - t0
            ax_t0.annotate(f'$\\Delta$={lead:.1f} h',
                           xy=(x_cases[i] + bar_w/2, max(tf, t0) + 0.15),
                           ha='center', fontsize=FONT_SMALL, color='darkred',
                           fontweight='bold')

    ax_t0.set_xticks(x_cases)
    ax_t0.set_xticklabels(cases, fontsize=FONT_MAIN)
    ax_t0.set_ylabel(r'Time from rain onset (h)   —   '
                     r'$t_\mathrm{fail}$ (gray) vs. $t_0$ (color)',
                     fontsize=FONT_MAIN)
    ax_t0.legend(fontsize=FONT_SMALL, handlelength=1.2, borderpad=0.5)
    ax_t0.grid(alpha=0.28, axis='y', lw=0.5)
    ax_t0.tick_params(labelsize=FONT_SMALL)

    out_d = os.path.join(output_dir, 'figB_d_t0_comparison.png')
    fig_d.savefig(out_d, dpi=DPI, bbox_inches='tight')
    plt.close(fig_d)
    print(f"  已保存单独 t₀ 对比图: {out_d}")

    # CSV — logistic_k → logistic_r
    rows = []
    for pct in pcts_avail:
        for case in cases:
            rs     = case_rs[case]
            t_fail = FAILURE_HOURS.get(case, np.nan)
            ef     = logistic_fits.get(case)
            for L_thr in L_thrs:
                t_DAS = np.nan
                for r in rs:
                    if r['largest_size'].get(pct, 0) >= L_thr:
                        t_DAS = r['hours']
                        break
                lead = t_fail - t_DAS if not np.isnan(t_DAS) else np.nan
                rows.append({
                    'case': case, 'percentile': pct,
                    'L_threshold_voxels': L_thr,
                    't_DAS_h': round(t_DAS, 3) if not np.isnan(t_DAS) else np.nan,
                    't_fail_h': t_fail,
                    'lead_time_h': round(lead, 3) if not np.isnan(lead) else np.nan,
                    'logistic_t0_h': round(ef['t0'], 4) if ef else np.nan,
                    'logistic_r':    round(ef['r'],  5) if ef else np.nan,
                    'logistic_K':    round(ef['K'],  2) if ef else np.nan,
                    'logistic_R2':   round(ef['r2'], 4) if ef else np.nan,
                })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'tableB_logistic_sensitivity.csv'), index=False)
    print(f"  Table CSV 已保存")


# ============================================================================
#  Figure C — 留一法 (LOO) 验证 (移除 (e) 子图, 整体改为 2 行)
# ============================================================================

def plot_figure_C_leave_one_out(results, logistic_fits, output_dir):
    print("\n[Fig C] 留一法阈值验证图 (含 Logistic LOO 拟合束)...")
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {c: sorted([r for r in results if r['case']==c],
                          key=lambda r: r['hours']) for c in cases}
    n_cases = len(cases)

    all_inside_global = np.concatenate([r['inside_vals'] for r in results])
    global_thr_base = {p: float(np.percentile(all_inside_global, p))
                       for p in THRESHOLD_PERCENTILES}

    loo_thresholds_all = []
    for i in range(len(results)):
        others = [r for j, r in enumerate(results) if j != i]
        if not others:
            loo_thresholds_all.append(global_thr_base)
            continue
        inside_loo = np.concatenate([r['inside_vals'] for r in others])
        loo_thresholds_all.append(
            {p: float(np.percentile(inside_loo, p)) for p in THRESHOLD_PERCENTILES})

    # ── 现在仅 2 行: 第 0 行 Lmax(t), 第 1 行 N(t); 右侧第 (n_cases) 列箱线图 ──
    row_h_C = 2.10
    fig_h_C = row_h_C * 2 + 0.4

    fig = plt.figure(figsize=(A4_W_IN, fig_h_C))
    gs = fig.add_gridspec(2, n_cases + 1,
                          hspace=0.55, wspace=0.42,
                          height_ratios=[1.0, 1.0],
                          width_ratios=[1] * n_cases + [1.15],
                          left=0.07, right=0.97, top=0.96, bottom=0.10)

    loo_thr_by_case = {c: [] for c in cases}

    for ci, case in enumerate(cases):
        rs   = case_rs[case]
        col  = CASE_COLORS.get(case, 'gray')
        idxs = [results.index(r) for r in rs]
        times = [r['hours'] for r in rs]

        lmax_base  = [r['largest_size'][MAIN_PCT] for r in rs]
        ncomp_base = [r['n_components'][MAIN_PCT] for r in rs]

        ax0 = fig.add_subplot(gs[0, ci])
        ax1 = fig.add_subplot(gs[1, ci])

        ef_base = logistic_fits.get(case)
        if ef_base is not None:
            ax0.plot(ef_base['t_dense'], ef_base['l_fit'],
                     color=col, lw=2.0, zorder=6,
                     label=f'{case}: Logistic (all)\n$R^2$={ef_base["r2"]:.3f}')

        loo_lmax_curves, loo_ncomp_curves = [], []
        for idx in idxs:
            loo_thr = loo_thresholds_all[idx]
            loo_thr_by_case[case].append(loo_thr[MAIN_PCT])

            lmax_c, ncomp_c = [], []
            for r in rs:
                cores, n = find_cores(r['V_masked'], r['slope_mask'],
                                      loo_thr[MAIN_PCT], min_size=1)
                lmax_c.append(cores[0]['size'] if cores else 0)
                ncomp_c.append(n)
            loo_lmax_curves.append(lmax_c)
            loo_ncomp_curves.append(ncomp_c)

            rs_loo = [r for r in rs if results.index(r) != idx]
            if len(rs_loo) >= 4:
                t_loo = np.array([r['hours'] for r in rs_loo])
                l_loo = np.array([r['largest_size'][MAIN_PCT] for r in rs_loo], float)
                ef_loo = fit_logistic_bootstrap(t_loo, l_loo, n_bootstrap=200)
                if ef_loo is not None:
                    ax0.plot(ef_loo['t_dense'], ef_loo['l_fit'],
                             color='gray', lw=0.8, alpha=0.38, zorder=2)

        for lmax_c in loo_lmax_curves:
            ax0.plot(times, lmax_c, color='gray', lw=0.8, alpha=0.32, zorder=1)
        ax0.plot(times, lmax_base, 'o', color=col, ms=6, zorder=7)

        for ncomp_c in loo_ncomp_curves:
            ax1.plot(times, ncomp_c, color='gray', lw=0.8, alpha=0.32, zorder=1)
        ax1.plot(times, ncomp_base, 's-', color=col, lw=1.8, ms=6,
                 zorder=5, label=f'{case}: baseline')

        if case in FAILURE_HOURS:
            for ax_ in [ax0, ax1]:
                ax_.axvline(FAILURE_HOURS[case], color=col, ls='--',
                            lw=1.2, alpha=0.40)

        ax0.set_xlabel(f'$t$ (h)  —  Case {case}', fontsize=FONT_MAIN)
        ax0.set_ylabel(r'$L_{\max}$ (voxels)', fontsize=FONT_MAIN)
        ax0.legend(fontsize=FONT_SMALL, handlelength=1.0, borderpad=0.4)
        ax0.grid(alpha=0.25, lw=0.5)
        ax0.tick_params(labelsize=FONT_SMALL)

        ax1.set_xlabel(f'$t$ (h)  —  Case {case}', fontsize=FONT_MAIN)
        ax1.set_ylabel(f'$N$ components (P{MAIN_PCT})', fontsize=FONT_MAIN)
        ax1.legend(fontsize=FONT_SMALL, handlelength=1.0, borderpad=0.4)
        ax1.grid(alpha=0.25, lw=0.5)
        ax1.tick_params(labelsize=FONT_SMALL)

    # ── 右侧箱线图 ──
    ax_b0 = fig.add_subplot(gs[0, n_cases])
    ax_b1 = fig.add_subplot(gs[1, n_cases])

    bp_data = [loo_thr_by_case[c] for c in cases]
    bp_cols = [CASE_COLORS.get(c, 'gray') for c in cases]

    bp0 = ax_b0.boxplot(bp_data, patch_artist=True, widths=0.45,
                        medianprops=dict(color='k', lw=2.0))
    for patch, col in zip(bp0['boxes'], bp_cols):
        patch.set_facecolor(col); patch.set_alpha(0.70)
    ax_b0.axhline(global_thr_base[MAIN_PCT], color='k', ls='--',
                  lw=1.4, label=f'Global P{MAIN_PCT}')
    ax_b0.set_xticks(range(1, len(cases)+1))
    ax_b0.set_xticklabels(cases, fontsize=FONT_MAIN)
    ax_b0.set_ylabel(f'P{MAIN_PCT} threshold value', fontsize=FONT_MAIN)
    ax_b0.legend(fontsize=FONT_SMALL, handlelength=1.0)
    ax_b0.grid(alpha=0.25, axis='y', lw=0.5)
    ax_b0.tick_params(labelsize=FONT_SMALL)

    rel_data = [[100*(v-global_thr_base[MAIN_PCT])/global_thr_base[MAIN_PCT]
                 for v in vals] for vals in bp_data]
    bp1 = ax_b1.boxplot(rel_data, patch_artist=True, widths=0.45,
                        medianprops=dict(color='k', lw=2.0))
    for patch, col in zip(bp1['boxes'], bp_cols):
        patch.set_facecolor(col); patch.set_alpha(0.70)
    ax_b1.axhline(0, color='k', ls='--', lw=1.4, label='0% (no change)')
    ax_b1.set_xticks(range(1, len(cases)+1))
    ax_b1.set_xticklabels(cases, fontsize=FONT_MAIN)
    ax_b1.set_ylabel('Relative threshold change (%)',
                     fontsize=FONT_MAIN)
    ax_b1.legend(fontsize=FONT_SMALL, handlelength=1.0)
    ax_b1.grid(alpha=0.25, axis='y', lw=0.5)
    ax_b1.tick_params(labelsize=FONT_SMALL)

    # 注: 原 (e) 子图已删除

    out_fig = os.path.join(output_dir, 'figC_logistic_leave_one_out.png')
    fig.savefig(out_fig, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_fig}")

    # CSV
    rows = []
    for case in cases:
        rs   = case_rs[case]
        idxs = [results.index(r) for r in rs]
        for r, idx in zip(rs, idxs):
            loo_thr  = loo_thresholds_all[idx]
            cores_l, n_l = find_cores(r['V_masked'], r['slope_mask'],
                                      loo_thr[MAIN_PCT], min_size=1)
            rows.append({
                'case': case, 'label': r['label'], 'hours': r['hours'],
                'baseline_lmax':  r['largest_size'][MAIN_PCT],
                'loo_lmax':       cores_l[0]['size'] if cores_l else 0,
                'baseline_ncomp': r['n_components'][MAIN_PCT],
                'loo_ncomp':      n_l,
                'baseline_threshold': global_thr_base[MAIN_PCT],
                'loo_threshold':      loo_thr[MAIN_PCT],
                'thr_rel_change_pct': round(
                    100*(loo_thr[MAIN_PCT]-global_thr_base[MAIN_PCT])
                    /global_thr_base[MAIN_PCT], 3),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'tableC_logistic_loo_validation.csv'), index=False)
    print(f"  Table CSV 已保存")


# ============================================================================
#  Figure D — 网格分辨率/插值方法敏感性图 (移除 titles)
# ============================================================================

def downsample_volume(V, xs, ys, zs, factor):
    if factor == 1:
        return V, xs, ys, zs
    return (V[::factor, ::factor, ::factor],
            xs[::factor], ys[::factor], zs[::factor])


def upsample_volume(V_ds, xs_ds, ys_ds, zs_ds,
                    xs_orig, ys_orig, zs_orig, method='linear'):
    interp = RegularGridInterpolator(
        (xs_ds, ys_ds, zs_ds), V_ds, method=method,
        bounds_error=False, fill_value=float(np.nanmin(V_ds)))
    pts = np.array(np.meshgrid(xs_orig, ys_orig, zs_orig,
                                indexing='ij')).reshape(3, -1).T
    return interp(pts).reshape(len(xs_orig), len(ys_orig), len(zs_orig))


def plot_figure_D_grid_sensitivity(results, global_thresholds, logistic_fits,
                                   output_dir):
    print("\n[Fig D] 网格分辨率/插值方法敏感性图 (含 Logistic 基准曲线)...")
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {c: sorted([r for r in results if r['case']==c],
                          key=lambda r: r['hours']) for c in cases}
    n_cases = len(cases)

    METHODS      = ['linear', 'nearest']
    METHOD_STYLE = {'linear': '-', 'nearest': '--'}
    METHOD_LABEL = {'linear': 'Linear', 'nearest': 'Nearest'}
    DS_LW        = {1: 2.0, 2: 1.4, 4: 0.9}
    DS_ALPHA     = {1: 1.0, 2: 0.78, 4: 0.52}
    DS_COLORS    = {
        (2, 'linear'):  '#ff7f0e',
        (2, 'nearest'): '#ffbb78',
        (4, 'linear'):  '#9467bd',
        (4, 'nearest'): '#c5b0d5',
    }

    row_h_D = 1.95
    fig_h_D = row_h_D * 3 + 0.4

    fig = plt.figure(figsize=(A4_W_IN, fig_h_D))
    gs = fig.add_gridspec(3, n_cases,
                          hspace=0.62, wspace=0.42,
                          left=0.08, right=0.96, top=0.96, bottom=0.07)

    for ci, case in enumerate(cases):
        rs      = case_rs[case]
        col     = CASE_COLORS.get(case, 'gray')
        times   = [r['hours'] for r in rs]
        is_left = (ci == 0)

        lmax_base  = [r['largest_size'][MAIN_PCT] for r in rs]
        ncomp_base = [r['n_components'][MAIN_PCT] for r in rs]

        ax0 = fig.add_subplot(gs[0, ci])
        ax1 = fig.add_subplot(gs[1, ci])
        ax2 = fig.add_subplot(gs[2, ci])

        ax0.axhline(0, color='k', lw=1.0, ls='--', alpha=0.55,
                    label='Baseline (0% error)')
        ax1.axhline(0, color='k', lw=1.0, ls='--', alpha=0.55,
                    label='Baseline')

        # 右轴 (线性刻度): Logistic 基准曲线 — 显示实际数值, 不再用 log
        ax0_r = ax0.twinx()
        ef_base = logistic_fits.get(case)
        if ef_base is not None:
            ax0_r.plot(ef_base['t_dense'], ef_base['l_fit'],
                       color=col, lw=1.8, ls=':', alpha=0.60,
                       label=f'{case} Logistic ($K$={ef_base["K"]:.0f})')
            ax0_r.scatter(ef_base['t_v'], ef_base['l_v'],
                          color=col, s=40, zorder=5, alpha=0.70)
            # 强制普通数字格式 (不用科学计数法, 不用 offset)
            sf = mticker.ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax0_r.yaxis.set_major_formatter(sf)
            if ci == n_cases - 1:
                ax0_r.set_ylabel(r'$L_{\max}$ (voxels)',
                                  fontsize=FONT_SMALL, color=col)
            ax0_r.tick_params(axis='y', labelcolor=col, labelsize=FONT_SMALL - 1)
            if is_left:
                ax0_r.legend(fontsize=FONT_SMALL - 1, loc='upper right',
                             handlelength=1.0, borderpad=0.4)

        # X-profile (最后时刻)
        r_last    = rs[-1]
        prof_orig = np.nanmean(r_last['V_masked'], axis=(1, 2))
        ax2.plot(r_last['xs'], prof_orig, color=col,
                 lw=DS_LW[1], alpha=DS_ALPHA[1], label=f'{case} Original')

        for factor in DOWNSAMPLE_FACTORS[1:]:
            for method in METHODS:
                ds_col = DS_COLORS.get((factor, method), 'gray')
                lbl    = f'DS\u00d7{factor} {METHOD_LABEL[method]}'

                # X-profile: 下采样 → 上采样 → 重新应用掩膜 → 剖面
                try:
                    V_in_r = np.nan_to_num(r_last['V_masked'], nan=0.0)
                    V_ds_r, xs_ds, ys_ds, zs_ds = downsample_volume(
                        V_in_r, r_last['xs'], r_last['ys'], r_last['zs'], factor)
                    V_up_r = upsample_volume(
                        V_ds_r, xs_ds, ys_ds, zs_ds,
                        r_last['xs'], r_last['ys'], r_last['zs'], method)
                    V_up_r_masked = np.where(r_last['slope_mask'], V_up_r, np.nan)
                    ax2.plot(r_last['xs'], np.nanmean(V_up_r_masked, axis=(1, 2)),
                             color=ds_col, lw=DS_LW[factor],
                             ls=METHOD_STYLE[method],
                             alpha=DS_ALPHA[factor], label=lbl)
                except Exception:
                    pass

                # Lmax / N 误差序列
                lmax_rel, ncomp_abs = [], []
                for r in rs:
                    try:
                        V_in = np.nan_to_num(r['V_masked'], nan=0.0)
                        V_ds_r, xs_ds, ys_ds, zs_ds = downsample_volume(
                            V_in, r['xs'], r['ys'], r['zs'], factor)
                        V_up = upsample_volume(
                            V_ds_r, xs_ds, ys_ds, zs_ds,
                            r['xs'], r['ys'], r['zs'], method)
                        V_up_m = np.where(r['slope_mask'], V_up, np.nan)
                        thr = global_thresholds[MAIN_PCT]
                        cores_up, n_up = find_cores(V_up_m, r['slope_mask'],
                                                    thr, min_size=1)
                        lmax_up = cores_up[0]['size'] if cores_up else 0
                        base_l  = max(r['largest_size'][MAIN_PCT], 1)
                        lmax_rel.append(100.0 * (lmax_up - base_l) / base_l)
                        ncomp_abs.append(n_up - r['n_components'][MAIN_PCT])
                    except Exception:
                        lmax_rel.append(np.nan)
                        ncomp_abs.append(np.nan)

                ax0.plot(times, lmax_rel,
                         color=ds_col, lw=DS_LW[factor],
                         ls=METHOD_STYLE[method],
                         alpha=DS_ALPHA[factor], label=lbl)
                ax1.plot(times, ncomp_abs,
                         color=ds_col, lw=DS_LW[factor],
                         ls=METHOD_STYLE[method],
                         alpha=DS_ALPHA[factor], label=lbl)

        if case in FAILURE_HOURS:
            for ax_ in [ax0, ax1]:
                ax_.axvline(FAILURE_HOURS[case], color=col,
                            ls=':', lw=1.0, alpha=0.40)

        # 轴标签 (每个子图的 case 信息融入 xlabel, 替代被删除的 title)
        ax0.set_xlabel(f'$t$ (h)  —  Case {case}', fontsize=FONT_MAIN)
        ax1.set_xlabel(f'$t$ (h)  —  Case {case}', fontsize=FONT_MAIN)
        ax2.set_xlabel(f'$X$ (cm)  —  Case {case}, last snapshot',
                       fontsize=FONT_MAIN)

        if is_left:
            ax0.set_ylabel(r'$L_{\max}$ relative error (%)', fontsize=FONT_MAIN)
            ax1.set_ylabel('$N(t)$ absolute error (count)', fontsize=FONT_MAIN)
            ax2.set_ylabel('Mean damage intensity', fontsize=FONT_MAIN)

        if is_left:
            ax0.legend(fontsize=FONT_SMALL - 1, loc='upper left',
                       handlelength=1.0, borderpad=0.4)
            ax1.legend(fontsize=FONT_SMALL - 1, loc='right',
                       handlelength=1.0, borderpad=0.2)
            ax2.legend(fontsize=FONT_SMALL - 1, loc='lower left',
                       handlelength=1.0, borderpad=0.4)

        ax0.grid(alpha=0.25, lw=0.5)
        ax0.tick_params(labelsize=FONT_SMALL)
        ax1.grid(alpha=0.25, lw=0.5)
        ax1.tick_params(labelsize=FONT_SMALL)
        ax2.axvline(SLOPE_CREST_LENGTH, color='gray', ls='--', alpha=0.40)
        ax2.grid(alpha=0.25, lw=0.5)
        ax2.tick_params(labelsize=FONT_SMALL)

    out_fig = os.path.join(output_dir, 'figD_logistic_grid_sensitivity.png')
    fig.savefig(out_fig, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {out_fig}")


# ============================================================================
#  主入口
# ============================================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  DAS 补充四图脚本  [主模型: Logistic, 增长速率参数 r]")
    print(f"  A4 宽度: {A4_W_IN:.3f} in | 单图宽: {PANEL_W:.3f} in | DPI: {DPI}")
    print("=" * 72)

    print("\n[Step 1] 构造全局阈值...")
    global_thresholds = compute_global_thresholds()
    if global_thresholds is None:
        print("  !! 所有文件不存在，退出")
        return
    print(f"  P{MAIN_PCT} 全局阈值 = {global_thresholds[MAIN_PCT]:.4f}")

    print("\n[Step 2] 加载快照数据...")
    results = load_all_results(global_thresholds)
    if len(results) < 4:
        print(f"  !! 有效快照 {len(results)} < 4")
        return
    print(f"  共加载 {len(results)} 个快照")

    print("\n[Step 3] Logistic 全局拟合 (各工况)...")
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {c: sorted([r for r in results if r['case']==c],
                          key=lambda r: r['hours']) for c in cases}
    logistic_fits = {}
    for case, rs in case_rs.items():
        t_arr = np.array([r['hours']                  for r in rs])
        l_arr = np.array([r['largest_size'][MAIN_PCT]  for r in rs], float)
        ef = fit_logistic_bootstrap(t_arr, l_arr)
        logistic_fits[case] = ef
        if ef:
            print(f"  {case}: K={ef['K']:.1f}  r={ef['r']:.4f} h⁻¹  "
                  f"t₀={ef['t0']:.3f} h  R²={ef['r2']:.4f}")
        else:
            print(f"  {case}: 拟合失败")

    print("\n[Step 4] Fig A — Bootstrap 参数 95% CI + 单独 Forest Plot")
    plot_figure_A_bootstrap_ci(results, OUTPUT_DIR)

    print("\n[Step 5] Fig B — 前兆时间敏感性热图 + 单独 t₀ 对比图")
    plot_figure_B_sensitivity_heatmap(results, logistic_fits, OUTPUT_DIR)

    print("\n[Step 6] Fig C — 留一法阈值验证 (无 (e) 子图)")
    plot_figure_C_leave_one_out(results, logistic_fits, OUTPUT_DIR)

    print("\n[Step 7] Fig D — 网格分辨率/插值方法敏感性")
    plot_figure_D_grid_sensitivity(results, global_thresholds,
                                   logistic_fits, OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("  全部完成！输出文件:")
    files = [
        ('figA_logistic_bootstrap_CI.png',      'Fig A 直方图主图 (A4宽, 无标题)'),
        ('figA_d_forest_plot.png',               'Fig A (d) Forest Plot (r/t0 量程已扩展)'),
        ('tableA_logistic_bootstrap_params.csv','Fig A 数据表 (列: r_per_h, ...)'),
        ('figB_logistic_sensitivity_heatmap.png','Fig B 热图主图 (颜色数字已同源)'),
        ('figB_d_t0_comparison.png',             'Fig B (d) t₀对比图'),
        ('tableB_logistic_sensitivity.csv',      'Fig B 数据表 (列: logistic_r)'),
        ('figC_logistic_leave_one_out.png',      'Fig C 主图 (无 (e) 子图)'),
        ('tableC_logistic_loo_validation.csv',   'Fig C 数据表'),
        ('figD_logistic_grid_sensitivity.png',   'Fig D 主图'),
    ]
    for fname, desc in files:
        print(f"  {fname:<52s} ← {desc}")
    print("=" * 72)


if __name__ == '__main__':
    main()
