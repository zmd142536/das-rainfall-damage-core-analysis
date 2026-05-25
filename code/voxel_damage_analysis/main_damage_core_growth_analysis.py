"""
================================================================================
 DAS损伤场分析 v2:修复版
================================================================================
 v1 的问题:
   1. N(t) 和最大核体积曲线把所有工况(S1/S2/S4)混在一条折线上
   2. 没有数据一致性检查(网格尺寸、值范围)
   3. 缺少"主核体积增长"的对数坐标图(真正的临滑前驱信号)

 v2 的改进:
   1. 自动按 case 分组绘图,每个工况独立曲线 + 一张总览
   2. 启动时强制检查所有CSV的网格尺寸和值范围,异常文件用 *** 标注
   3. 增加 largest_size(t) 的对数坐标图(论文核心证据)
   4. 增加主核位置迁移轨迹图(按工况分)

 用法和 v1 一样,只改顶部 USER CONFIG 里的 CSV_FILES 和 OUTPUT_DIR。
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import curve_fit as _scipy_curve_fit
from scipy.stats import t as _t_dist
from matplotlib.lines import Line2D

# ============================================================================
#  USER CONFIG
# ============================================================================

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
    (r'', 'S4', 'T_20:30', 11.5),
]

# 各工况的失稳时刻(用于在演化图上画竖线标注)
FAILURE_HOURS = {
    'S1': 8.63,
    'S2': 11.61,
    'S4': 11.50,
}

OUTPUT_DIR =  r'' 

# 坡体几何
SLOPE_CREST_LENGTH = 100
SLOPE_TOE_LENGTH   = 180
SLOPE_HEIGHT       = 100

# 分析参数
THRESHOLD_PERCENTILES = [70, 75, 80, 85, 90, 92, 95, 97]
MAIN_PCT = 90
MIN_COMP_VOXELS = 5
TOP_N_CORES = 6

IS_LOG = True
USE_GLOBAL_THRESHOLD = True

# 数据一致性检查
EXPECTED_GRID = (50, 50, 50)   # 期望网格大小
VMAX_OUTLIER_FACTOR = 5.0       # vmax 超过中位数 N 倍则警告

# 工况配色
CASE_COLORS = {'S1': '#d62728', 'S2': '#1f77b4', 'S4': '#2ca02c'}

# ============================================================================


def load_voxler_csv(path):
    df = pd.read_csv(path, header=None, names=['x','y','z','v'])
    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    zs = np.sort(df['z'].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx*ny*nz != len(df):
        raise ValueError(f"网格不匹配:nx*ny*nz={nx*ny*nz} 但行数={len(df)}")
    df_sorted = df.sort_values(['x','y','z']).reset_index(drop=True)
    V = df_sorted['v'].values.reshape(nx, ny, nz)
    return V, xs, ys, zs


def build_slope_mask(xs, ys, zs):
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing='ij')
    slope_dx = SLOPE_TOE_LENGTH - SLOPE_CREST_LENGTH
    x_max_at_z = SLOPE_TOE_LENGTH - (slope_dx / SLOPE_HEIGHT) * Z3
    mask = (Z3 >= 0) & (Z3 <= SLOPE_HEIGHT) & (X3 >= 0) & (X3 <= x_max_at_z)
    return mask, X3, Y3, Z3


def find_cores(V_masked, slope_mask, X3, Y3, Z3, threshold, min_size=5):
    high = (V_masked > threshold) & slope_mask
    structure = np.ones((3,3,3), dtype=int)
    labeled, ncomp = ndimage.label(high, structure=structure)
    cores = []
    if ncomp > 0:
        for cid in range(1, ncomp + 1):
            cmask = (labeled == cid)
            sz = int(cmask.sum())
            if sz < min_size:
                continue
            cores.append({
                'size': sz,
                'cx': float(X3[cmask].mean()),
                'cy': float(Y3[cmask].mean()),
                'cz': float(Z3[cmask].mean()),
                'vmax': float(V_masked[cmask].max()),
                'vmean': float(V_masked[cmask].mean()),
            })
    cores.sort(key=lambda c: c['size'], reverse=True)
    return cores, ncomp


def compute_profiles(V_masked, xs, ys, zs, high_thr=None):
    nx, ny, nz = V_masked.shape
    prof_x = np.zeros(nx)
    prof_z = np.zeros(nz)
    high_frac_x = np.zeros(nx)
    for i in range(nx):
        sl = V_masked[i,:,:]
        v = sl[~np.isnan(sl)]
        if len(v) > 0:
            prof_x[i] = v.mean()
            if high_thr is not None:
                high_frac_x[i] = (v > high_thr).sum() / len(v)
    for k in range(nz):
        sl = V_masked[:,:,k]
        v = sl[~np.isnan(sl)]
        if len(v) > 0:
            prof_z[k] = v.mean()
    return prof_x, prof_z, high_frac_x


def consistency_check(csv_files):
    """对所有 CSV 做一致性检查,异常文件标 *** 在终端输出。
    返回每个文件的诊断字典。
    """
    print("="*80)
    print(" 数据一致性检查")
    print("="*80)
    diagnostics = []
    all_vmax = []
    all_grid_sizes = []
    for path, case, label, hours in csv_files:
        if not os.path.exists(path):
            print(f"  ! 缺失: {path}")
            diagnostics.append(None)
            continue
        try:
            V, xs, ys, zs = load_voxler_csv(path)
            grid = (len(xs), len(ys), len(zs))
            slope_mask, _, _, _ = build_slope_mask(xs, ys, zs)
            inside = V[slope_mask]
            d = {
                'path': path, 'case': case, 'label': label, 'hours': hours,
                'grid': grid, 'inside_count': int(slope_mask.sum()),
                'vmax': float(inside.max()), 'vmedian': float(np.median(inside)),
                'vmin': float(inside.min()),
            }
            diagnostics.append(d)
            all_vmax.append(d['vmax'])
            all_grid_sizes.append(grid)
        except Exception as e:
            print(f"  ! 读取失败 {path}: {e}")
            diagnostics.append(None)

    if not diagnostics:
        return diagnostics

    vmax_median = np.median(all_vmax)
    print(f"\n  共 {len([d for d in diagnostics if d])} 个文件,vmax 中位数 = {vmax_median:.2f}")
    print(f"  期望网格 = {EXPECTED_GRID}")
    print()
    print(f"  {'工况':>4s} {'标签':>10s} {'时间':>6s} {'网格':>15s} "
          f"{'坡内':>7s} {'vmax':>9s} {'中位':>8s}  状态")
    print("  " + "-"*78)
    for d in diagnostics:
        if d is None:
            continue
        flags = []
        if d['grid'] != EXPECTED_GRID:
            flags.append('***网格异常')
        if d['vmax'] > vmax_median * VMAX_OUTLIER_FACTOR:
            flags.append(f'***vmax异常({d["vmax"]/vmax_median:.1f}x中位数)')
        flag_str = ' '.join(flags) if flags else 'OK'
        print(f"  {d['case']:>4s} {d['label']:>10s} {d['hours']:>5.2f}h "
              f"{str(d['grid']):>15s} {d['inside_count']:>7d} "
              f"{d['vmax']:>9.3f} {d['vmedian']:>8.3f}  {flag_str}")
    print()
    abnormal = [d for d in diagnostics if d and (
        d['grid'] != EXPECTED_GRID or d['vmax'] > vmax_median * VMAX_OUTLIER_FACTOR
    )]
    if abnormal:
        print(f"  !! 发现 {len(abnormal)} 个异常文件,建议先修复后再分析")
        print(f"     (脚本会继续运行,但这些文件的结果不可信)")
    print("="*80 + "\n")
    return diagnostics


def analyze_snapshot(path, case, label, hours, global_thresholds=None):
    V, xs, ys, zs = load_voxler_csv(path)
    slope_mask, X3, Y3, Z3 = build_slope_mask(xs, ys, zs)
    V_masked = np.where(slope_mask, V, np.nan)
    inside = V_masked[~np.isnan(V_masked)]

    if global_thresholds is None:
        thr_map = {p: float(np.percentile(inside, p)) for p in THRESHOLD_PERCENTILES}
    else:
        thr_map = global_thresholds

    n_components = {}
    largest_size = {}
    for p, thr in thr_map.items():
        cores, n = find_cores(V_masked, slope_mask, X3, Y3, Z3, thr, min_size=1)
        n_components[p] = n
        largest_size[p] = cores[0]['size'] if cores else 0

    main_thr = thr_map[MAIN_PCT]
    cores, _ = find_cores(V_masked, slope_mask, X3, Y3, Z3, main_thr,
                          min_size=MIN_COMP_VOXELS)
    prof_x, prof_z, high_frac_x = compute_profiles(V_masked, xs, ys, zs,
                                                    high_thr=main_thr)
    return {
        'path': path, 'case': case, 'label': label, 'hours': hours,
        'V_masked': V_masked, 'slope_mask': slope_mask,
        'xs': xs, 'ys': ys, 'zs': zs, 'inside_vals': inside,
        'thr_map': thr_map, 'n_components': n_components,
        'largest_size': largest_size, 'cores': cores,
        'prof_x': prof_x, 'prof_z': prof_z, 'high_frac_x': high_frac_x,
        'main_thr': main_thr,
    }


def plot_evolution_by_case(results, save_path):
    """v2 版的核心图:按工况分组绘制,且新增对数坐标的主核体积增长图。"""
    if len(results) < 2:
        print("[evolution] 至少需要2个时刻")
        return
    cases = sorted(set(r['case'] for r in results))
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # (a) N(t) 按工况分组 —— 线性
    ax = fig.add_subplot(gs[0, 0])
    for case in cases:
        rs = sorted([r for r in results if r['case']==case], key=lambda r: r['hours'])
        if not rs: continue
        times = [r['hours'] for r in rs]
        ns = [r['n_components'][MAIN_PCT] for r in rs]
        col = CASE_COLORS.get(case, 'gray')
        ax.plot(times, ns, 'o-', color=col, lw=2, ms=8, label=case)
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col, ls='--', alpha=0.4)
    ax.set_xlabel('Time since rain onset (h)')
    ax.set_ylabel(f'# components at P{MAIN_PCT}')
    ax.set_title('(a) Component count over time, BY CASE\n(dashed line = failure time)',
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) 最大核体积 —— 线性
    ax = fig.add_subplot(gs[0, 1])
    for case in cases:
        rs = sorted([r for r in results if r['case']==case], key=lambda r: r['hours'])
        if not rs: continue
        times = [r['hours'] for r in rs]
        ls = [r['largest_size'][MAIN_PCT] for r in rs]
        col = CASE_COLORS.get(case, 'gray')
        ax.plot(times, ls, 's-', color=col, lw=2, ms=8, label=case)
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col, ls='--', alpha=0.4)
    ax.set_xlabel('Time since rain onset (h)')
    ax.set_ylabel(f'Largest component size (voxels)')
    ax.set_title('(b) Largest core size over time, BY CASE\n(linear scale)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) ★ 关键图:最大核体积 对数坐标 ★
    ax = fig.add_subplot(gs[0, 2])
    for case in cases:
        rs = sorted([r for r in results if r['case']==case], key=lambda r: r['hours'])
        if not rs: continue
        times = [r['hours'] for r in rs]
        ls = [max(r['largest_size'][MAIN_PCT], 1) for r in rs]  # 1 防止 log0
        col = CASE_COLORS.get(case, 'gray')
        ax.semilogy(times, ls, 'D-', color=col, lw=2, ms=8, label=case)
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col, ls='--', alpha=0.4)
    ax.set_xlabel('Time since rain onset (h)')
    ax.set_ylabel(f'Largest core size (voxels, LOG)')
    ax.set_title('(c) ★ Main-core EXPONENTIAL growth\n(this is the key precursor signal)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

    # (d) X方向剖面演化 —— 每个工况单独子图
    for ci, case in enumerate(cases[:3]):
        ax = fig.add_subplot(gs[1, ci])
        rs = sorted([r for r in results if r['case']==case], key=lambda r: r['hours'])
        if not rs: continue
        cmap_t = plt.cm.viridis(np.linspace(0, 0.9, len(rs)))
        for r, col in zip(rs, cmap_t):
            ax.plot(r['xs'], r['prof_x'], color=col, lw=1.8,
                    label=f"{r['label']} ({r['hours']}h)")
        ax.axvline(SLOPE_CREST_LENGTH, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('X (cm)'); ax.set_ylabel('Mean damage intensity')
        ax.set_title(f'(d{ci+1}) Case {case}: X-profile evolution', fontsize=10)
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)

    # (e) 主核位置迁移 —— 每个工况单独子图
    for ci, case in enumerate(cases[:3]):
        ax = fig.add_subplot(gs[2, ci])
        rs = sorted([r for r in results if r['case']==case], key=lambda r: r['hours'])
        if not rs: continue
        cmap_t = plt.cm.viridis(np.linspace(0, 0.9, len(rs)))
        # 画前 TOP_N_CORES 个核
        for r, col in zip(rs, cmap_t):
            for i, c in enumerate(r['cores'][:TOP_N_CORES]):
                ms = max(np.sqrt(c['size']) * 1.2, 5)
                ax.scatter(c['cx'], c['cz'], s=ms, color=col, alpha=0.6,
                           edgecolors='k', lw=0.5)
        # 主核迁移轨迹(rank 1)
        traj_x = [r['cores'][0]['cx'] for r in rs if r['cores']]
        traj_z = [r['cores'][0]['cz'] for r in rs if r['cores']]
        if len(traj_x) > 1:
            ax.plot(traj_x, traj_z, 'k-', lw=1.2, alpha=0.7)
            ax.plot(traj_x[0], traj_z[0], 'go', ms=12, label='start')
            ax.plot(traj_x[-1], traj_z[-1], 'r*', ms=15, label='end')
        ax.plot([0,SLOPE_CREST_LENGTH,SLOPE_TOE_LENGTH,0,0],
                [SLOPE_HEIGHT,SLOPE_HEIGHT,0,0,SLOPE_HEIGHT], 'k-', lw=1.2)
        ax.set_xlabel('X (cm)'); ax.set_ylabel('Z (cm)')
        ax.set_title(f'(e{ci+1}) Case {case}: main-core trajectory', fontsize=10)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlim(-5, SLOPE_TOE_LENGTH+5); ax.set_ylim(-5, SLOPE_HEIGHT+10)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('Damage organization evolution — by case\n'
                 '(main-core growth on log scale is the cleanest precursor)',
                 fontsize=13, fontweight='bold', y=0.995)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")


def export_csv_results(results, output_dir):
    rows = []
    for r in results:
        for p, n in r['n_components'].items():
            rows.append({
                'case': r['case'], 'label': r['label'], 'hours': r['hours'],
                'percentile': p, 'threshold': r['thr_map'][p],
                'n_components': n, 'largest_size': r['largest_size'][p],
            })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'n_components_table.csv'),
                              index=False)
    rows = []
    for r in results:
        for i, c in enumerate(r['cores']):
            rows.append({
                'case': r['case'], 'label': r['label'], 'hours': r['hours'],
                'rank': i+1, 'size': c['size'],
                'cx': c['cx'], 'cy': c['cy'], 'cz': c['cz'],
                'vmax': c['vmax'], 'vmean': c['vmean'],
            })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'cores_table.csv'),
                              index=False)
    print(f"[csv] 已导出结果表到 {output_dir}")


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ★ 第0步:数据一致性检查 ★
    diagnostics = consistency_check(CSV_FILES)

    # 第一遍:全局阈值
    if USE_GLOBAL_THRESHOLD:
        print("[第1遍] 聚合所有时刻的值分布构造全局阈值...")
        all_inside = []
        for path, case, label, hours in CSV_FILES:
            if not os.path.exists(path): continue
            V, xs, ys, zs = load_voxler_csv(path)
            slope_mask, _, _, _ = build_slope_mask(xs, ys, zs)
            all_inside.append(V[slope_mask])
        all_inside = np.concatenate(all_inside)
        global_thresholds = {p: float(np.percentile(all_inside, p))
                             for p in THRESHOLD_PERCENTILES}
        print(f"  全局阈值: {global_thresholds}\n")
    else:
        global_thresholds = None

    # 第二遍:逐时刻分析
    print("[第2遍] 逐时刻分析...")
    results = []
    for path, case, label, hours in CSV_FILES:
        if not os.path.exists(path): continue
        print(f"  > {case} {label} ({hours} h)")
        r = analyze_snapshot(path, case, label, hours,
                             global_thresholds=global_thresholds)
        results.append(r)

    # 第三遍:出图 + CSV
    print("\n[第3遍] 跨时刻演化(按工况分组)...")
    if len(results) >= 2:
        evopath = os.path.join(OUTPUT_DIR, 'evolution_by_case.png')
        plot_evolution_by_case(results, evopath)
    export_csv_results(results, OUTPUT_DIR)

    print("\n完成。重点看:")
    print("  1. 终端的'数据一致性检查'输出 —— 异常文件先处理")
    print("  2. evolution_by_case.png 的 (c) 子图 —— 主核体积对数增长")


if __name__ == '__main__':
    main()
# ===========================================================================
#  1. 指数拟合 (log10-线性回归) + bootstrap CI
# ===========================================================================

def fit_exponential_log(times, lmax, n_bootstrap=2000, rng_seed=42):
    """
    对 log10(L_max) ~ t 做 OLS 线性回归, 估计:
        k   : 自然对数底增长率 (h⁻¹)
        T_d : 倍增时间 = ln2/k (h)
    CI 策略:
        - 均值预测 95% CI 带 : 解析公式 (用于图中阴影)
        - k / T_d 点估计 CI  : bootstrap 百分位 (用于 forest plot)
    返回 dict 或 None (点数 < 3)
    """
    np.random.seed(rng_seed)
    t = np.asarray(times, float)
    l = np.asarray(lmax,  float)
    ok = l > 0
    t, l = t[ok], l[ok]
    if len(t) < 3:
        return None

    log_l = np.log10(l)
    coeffs, _ = np.polyfit(t, log_l, 1, cov=True)
    k10  = float(coeffs[0])           # slope in log10 space
    k_ln = k10 * np.log(10)           # natural-log rate  (h⁻¹)
    T_d  = np.log(2) / k_ln if k_ln > 0 else np.nan

    log_pred = np.polyval(coeffs, t)
    ss_res   = float(np.sum((log_l - log_pred) ** 2))
    ss_tot   = float(np.sum((log_l - log_l.mean()) ** 2))
    r2_log   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # --- 解析 95% CI 带 (mean-prediction interval in log10 space) ---
    n       = len(t)
    df_reg  = max(n - 2, 1)
    t_crit  = float(_t_dist.ppf(0.975, df_reg))
    t_mean  = float(t.mean())
    Sxx     = float(np.sum((t - t_mean) ** 2)) + 1e-14
    s2      = max(ss_res / df_reg, 1e-14)          # MSE

    def ci_band(t_new):
        """返回 (lower, upper) 在原始 L 空间的预测均值 95% CI"""
        t_new = np.asarray(t_new, float)
        y_hat = np.polyval(coeffs, t_new)
        se    = np.sqrt(s2 * (1.0 / n + (t_new - t_mean) ** 2 / Sxx))
        return 10 ** (y_hat - t_crit * se), 10 ** (y_hat + t_crit * se)

    # --- Bootstrap CI on k_ln, T_d ---
    k_bs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if np.unique(t[idx]).size < 2:
            continue
        try:
            cb = np.polyfit(t[idx], log_l[idx], 1)
            k_bs.append(float(cb[0]) * np.log(10))
        except Exception:
            pass
    k_bs = np.array(k_bs)
    if len(k_bs) >= 20:
        k_ci  = (float(np.percentile(k_bs, 2.5)),
                 float(np.percentile(k_bs, 97.5)))
        Td_lo = np.log(2) / k_ci[1] if k_ci[1] > 0 else np.nan
        Td_hi = np.log(2) / k_ci[0] if k_ci[0] > 0 else np.nan
        Td_ci = (Td_lo, Td_hi)
    else:
        k_ci  = (np.nan, np.nan)
        Td_ci = (np.nan, np.nan)

    return dict(
        k=k_ln, k10=k10, T_d=T_d,
        k_ci=k_ci, Td_ci=Td_ci, r2_log=r2_log,
        coeffs=coeffs, t_v=t, l_v=l, log_l=log_l,
        log_pred=log_pred, ci_band=ci_band,
    )


# ===========================================================================
#  2. 四模型拟合 + R² / AICc / BIC
# ===========================================================================

def fit_all_models(times, lmax):
    """
    拟合四种模型:
        Exponential  L = A·exp(k·t)       OLS in log space  (2 params)
        Linear       L = a·t + b           OLS               (2 params)
        Power-law    L = a·t^b             OLS in log-log    (2 params, t>0)
        Logistic     L = K/(1+exp(-k(t-t0))) nonlinear       (3 params)

    统计量 R², AICc, BIC 均在原始 L 空间计算 (正态误差假设).
    返回 dict[model_name -> stats_and_curve_dict]
    """
    t = np.asarray(times, float)
    l = np.asarray(lmax,  float)
    ok = l > 0
    t, l = t[ok], l[ok]
    n = len(t)
    if n < 3:
        return {}

    t_dense = np.linspace(t.min(), t.max(), 200)

    def _info_stats(obs, pred, n_p):
        """R², AICc, BIC — 原始空间, 正态残差"""
        ssr  = float(np.sum((obs - pred) ** 2))
        sst  = float(np.sum((obs - obs.mean()) ** 2))
        r2   = 1.0 - ssr / sst if sst > 0 else np.nan
        sig2 = max(ssr / n, 1e-14)
        ll   = -n / 2.0 * np.log(2 * np.pi * sig2) - ssr / (2 * sig2)
        aic  = 2 * n_p - 2 * ll
        aicc = aic + 2 * n_p * (n_p + 1) / max(n - n_p - 1, 1)
        bic  = n_p * np.log(n) - 2 * ll
        return r2, aicc, bic

    out = {}

    # ---- Exponential ----
    c     = np.polyfit(t, np.log(l), 1)
    Ae, ke = float(np.exp(c[1])), float(c[0])
    pe    = Ae * np.exp(ke * t)
    r2, aicc, bic = _info_stats(l, pe, 2)
    out['Exponential'] = dict(
        r2=r2, aicc=aicc, bic=bic, k=ke, A=Ae, n_params=2,
        t_plot=t_dense, l_plot=Ae * np.exp(ke * t_dense),
    )

    # ---- Linear ----
    cl    = np.polyfit(t, l, 1)
    pl    = np.polyval(cl, t)
    r2, aicc, bic = _info_stats(l, pl, 2)
    out['Linear'] = dict(
        r2=r2, aicc=aicc, bic=bic, coeffs=cl, n_params=2,
        t_plot=t_dense, l_plot=np.polyval(cl, t_dense),
    )

    # ---- Power-law (t > 0 only) ----
    vp = t > 0
    if vp.sum() >= 3:
        tp, lp  = t[vp], l[vp]
        cp      = np.polyfit(np.log(tp), np.log(lp), 1)
        ap, bp  = float(np.exp(cp[1])), float(cp[0])
        pp      = ap * tp ** bp
        r2, aicc, bic = _info_stats(lp, pp, 2)
        t_dp    = np.linspace(max(tp.min(), 1e-2), tp.max(), 200)
        out['Power-law'] = dict(
            r2=r2, aicc=aicc, bic=bic, a=ap, b=bp, n_params=2,
            t_plot=t_dp, l_plot=ap * t_dp ** bp,
        )

    # ---- Logistic ----
    try:
        def _logistic(t_, K, k, t0):
            return K / (1.0 + np.exp(-k * (t_ - t0)))
        p0     = [l.max() * 2.0, 0.5, float(t.mean())]
        popt, _ = _scipy_curve_fit(
            _logistic, t, l, p0=p0, maxfev=12000,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        )
        pg     = _logistic(t, *popt)
        r2, aicc, bic = _info_stats(l, pg, 3)
        out['Logistic'] = dict(
            r2=r2, aicc=aicc, bic=bic,
            K=float(popt[0]), k=float(popt[1]), t0=float(popt[2]),
            n_params=3,
            t_plot=t_dense, l_plot=_logistic(t_dense, *popt),
        )
    except Exception:
        pass

    return out


# ===========================================================================
#  3. Figure 5 — 六宫格主图
# ===========================================================================

def plot_figure5_lmax_growth(results, output_dir):
    """
    Figure 5 (2×3 grid):
    (a) L_max(t) 线性坐标  — 三工况原始曲线
    (b) log10 L_max(t)     — 指数拟合线 + 95% CI 阴影
    (c) Forest Plot        — k 和 T_d 及其 bootstrap 95% CI
    (d) 四模型拟合对比     — 代表工况, 含 R²/AICc/BIC
    (e) 阈值敏感性         — P85 / P90 / P95
    (f) t_DAS 判据敏感性   — L_thr=50/100/200 体素下的预警时间
    """
    cases   = sorted(set(r['case'] for r in results))
    case_rs = {
        c: sorted([r for r in results if r['case'] == c],
                  key=lambda r: r['hours'])
        for c in cases
    }

    # ---- 预计算拟合 ----
    exp_fits   = {}
    model_fits = {}
    for case, rs in case_rs.items():
        t_arr = np.array([r['hours'] for r in rs])
        l_arr = np.array([r['largest_size'][MAIN_PCT] for r in rs], float)
        exp_fits[case]   = fit_exponential_log(t_arr, l_arr)
        model_fits[case] = fit_all_models(t_arr, l_arr)

    # ---- 画布 ----
    fig = plt.figure(figsize=(19, 11.5))
    gs  = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.36)

    # ─────────────────────────────────────────────────────── (a) 线性坐标
    ax = fig.add_subplot(gs[0, 0])
    for case in cases:
        rs    = case_rs[case]
        t_pts = [r['hours'] for r in rs]
        l_pts = [r['largest_size'][MAIN_PCT] for r in rs]
        col   = CASE_COLORS.get(case, 'gray')
        ax.plot(t_pts, l_pts, 'o-', color=col, lw=2.2, ms=9,
                label=case, zorder=3)
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col, ls='--',
                       lw=1.5, alpha=0.40, label='_nolegend_')
    ax.set_xlabel('t  (h from rain onset)', fontsize=11)
    ax.set_ylabel(r'$L_{\max}(t)$  (voxels)', fontsize=11)
    ax.set_title('(a)  Main-core size — linear scale', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.30)

    # ─────────────────────────────────────────────────────── (b) 对数坐标 + 拟合
    ax = fig.add_subplot(gs[0, 1])
    for case in cases:
        rs    = case_rs[case]
        t_pts = np.array([r['hours'] for r in rs])
        l_pts = np.array([r['largest_size'][MAIN_PCT] for r in rs], float)
        col   = CASE_COLORS.get(case, 'gray')
        valid = l_pts > 0
        ax.semilogy(t_pts[valid], l_pts[valid], 'o', color=col, ms=9, zorder=5)
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col, ls='--',
                       lw=1.5, alpha=0.40)
        ef = exp_fits.get(case)
        if ef is None:
            continue
        t_rng  = np.linspace(ef['t_v'].min(), ef['t_v'].max(), 200)
        l_fit  = 10 ** np.polyval(ef['coeffs'], t_rng)
        lb, ub = ef['ci_band'](t_rng)
        k_str  = f"k={ef['k']:.3f} h⁻¹"
        Td_str = f"$T_d$={ef['T_d']:.2f} h"
        R2_str = f"$R^2$={ef['r2_log']:.2f}"
        ax.semilogy(t_rng, l_fit, '-', color=col, lw=2.8,
                    label=f"{case}: {k_str},  {Td_str},  {R2_str}")
        ax.fill_between(t_rng,
                         np.clip(lb, 0.5, None),
                         np.clip(ub, 0.5, None),
                         color=col, alpha=0.13)
    ax.set_xlabel('t  (h from rain onset)', fontsize=11)
    ax.set_ylabel(r'$L_{\max}(t)$  (voxels,  log scale)', fontsize=11)
    ax.set_title('(b)  Exponential growth (log scale)\n'
                 'Solid = fit,  shaded = 95% CI mean-prediction',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.30, which='both')

    # ─────────────────────────────────────────────────────── (c) Forest plot
    ax = fig.add_subplot(gs[0, 2])
    ax2 = ax.twiny()
    y   = np.arange(len(cases))

    for i, case in enumerate(cases):
        col = CASE_COLORS.get(case, 'gray')
        ef  = exp_fits.get(case)
        if ef is None:
            continue

        k_val   = ef['k']
        Td_val  = ef['T_d']
        k_lo_e  = k_val  - ef['k_ci'][0]   if not np.isnan(ef['k_ci'][0])  else 0.0
        k_hi_e  = ef['k_ci'][1]  - k_val   if not np.isnan(ef['k_ci'][1])  else 0.0
        Td_lo_e = Td_val - ef['Td_ci'][0]  if not np.isnan(ef['Td_ci'][0]) else 0.0
        Td_hi_e = ef['Td_ci'][1] - Td_val  if not np.isnan(ef['Td_ci'][1]) else 0.0

        # k on primary (bottom) x-axis
        ax.errorbar(k_val, y[i] + 0.18,
                     xerr=[[k_lo_e], [k_hi_e]],
                     fmt='s', color=col, ms=12, capsize=6, lw=2.2,
                     label=f'{case}: k={k_val:.3f} h⁻¹')
        # T_d on secondary (top) x-axis
        ax2.errorbar(Td_val, y[i] - 0.18,
                      xerr=[[Td_lo_e], [Td_hi_e]],
                      fmt='D', color=col, ms=12, capsize=6, lw=2.2,
                      alpha=0.68, linestyle='dashed',
                      label=f'{case}: $T_d$={Td_val:.2f} h')

    ax.set_yticks(y)
    ax.set_yticklabels(cases, fontsize=12)
    ax.set_xlabel('k  (h⁻¹)  [■ solid]', fontsize=11)
    ax2.set_xlabel('$T_d$  (h)  [◆ dashed]', fontsize=10, color='dimgray')
    ax.set_title('(c)  k  and  $T_d$  with  95% CI\n'
                 '(bootstrap, 2000 resamples)',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.30, axis='x')
    ax.legend(fontsize=8.5, loc='lower right')

    # ─────────────────────────────────────────────────────── (d) 四模型对比
    # 默认选 S2 作代表工况;不存在则取第一个
    rep_case = 'S2' if 'S2' in cases else cases[0]
    ax = fig.add_subplot(gs[1, 0])
    rs_rep  = case_rs[rep_case]
    t_data  = np.array([r['hours'] for r in rs_rep])
    l_data  = np.array([r['largest_size'][MAIN_PCT] for r in rs_rep], float)
    ax.scatter(t_data, l_data, s=90, color='k', zorder=7,
               label='Data', marker='o')

    m_cfg = {                                       # name → (color, ls, lw)
        'Exponential': ('tomato',     '-',   3.0),
        'Linear':      ('steelblue',  '--',  2.2),
        'Power-law':   ('seagreen',   '-.',  2.2),
        'Logistic':    ('darkorange', ':',   3.0),
    }
    mf = model_fits.get(rep_case, {})
    for mname, (mc, mls, mlw) in m_cfg.items():
        md = mf.get(mname)
        if md is None:
            continue
        r2_s   = f"$R^2$={md['r2']:.3f}"
        aicc_s = f"AICc={md['aicc']:.1f}"
        bic_s  = f"BIC={md['bic']:.1f}"
        ax.plot(md['t_plot'], md['l_plot'],
                color=mc, ls=mls, lw=mlw,
                label=f"{mname}\n{r2_s}  {aicc_s}  {bic_s}")
    ax.set_xlabel('t  (h from rain onset)', fontsize=11)
    ax.set_ylabel(r'$L_{\max}(t)$  (voxels)', fontsize=11)
    ax.set_title(f'(d)  Four-model comparison — Case {rep_case}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.2, loc='upper left')
    ax.grid(alpha=0.30)

    # ─────────────────────────────────────────────────────── (e) 阈值敏感性
    ax = fig.add_subplot(gs[1, 1])
    sens_pcts   = [p for p in [85, 90, 95] if p in THRESHOLD_PERCENTILES]
    pct_styles  = ['-', '--', '-.']
    pct_alphas  = [1.00, 0.85, 0.70]
    pct_lws     = [2.6, 2.0, 1.6]

    for case in cases:
        rs  = case_rs[case]
        col = CASE_COLORS.get(case, 'gray')
        for pi, pct in enumerate(sens_pcts):
            t_pts = [r['hours'] for r in rs]
            l_pts = [r['largest_size'].get(pct, 0) for r in rs]
            ax.plot(t_pts, l_pts,
                     ls=pct_styles[pi], color=col,
                     lw=pct_lws[pi], alpha=pct_alphas[pi])
        if case in FAILURE_HOURS:
            ax.axvline(FAILURE_HOURS[case], color=col,
                       ls=':', lw=1.2, alpha=0.35)

    # 组合图例: 工况颜色 + 阈值线型
    leg_h = (
        [Line2D([0],[0], color=CASE_COLORS.get(c,'gray'), lw=2.4, label=c)
         for c in cases]
        + [Line2D([0],[0], color='k', ls=pct_styles[i], lw=pct_lws[i],
                  alpha=pct_alphas[i], label=f'P{p}')
           for i, p in enumerate(sens_pcts)]
    )
    ax.legend(handles=leg_h, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_xlabel('t  (h from rain onset)', fontsize=11)
    ax.set_ylabel(r'$L_{\max}(t)$  (voxels)', fontsize=11)
    ax.set_title('(e)  Threshold sensitivity\n'
                 '(P85 / P90 / P95, dashed = failure time)',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.30)

    # ─────────────────────────────────────────────────────── (f) t_DAS 预警时间
    ax = fig.add_subplot(gs[1, 2])
    L_thresholds  = [50, 100, 200]
    bar_w         = 0.24
    x_idx         = np.arange(len(cases))
    bar_colors    = ['#f5c842', '#e07b28', '#b33015']

    for bi, L_thr in enumerate(L_thresholds):
        lead_times = []
        for case in cases:
            rs     = case_rs[case]
            t_fail = FAILURE_HOURS.get(case, np.nan)
            t_DAS  = np.nan
            for r in rs:
                if r['largest_size'].get(MAIN_PCT, 0) >= L_thr:
                    t_DAS = r['hours']
                    break
            lead = (t_fail - t_DAS
                    if (not np.isnan(t_DAS) and not np.isnan(t_fail))
                    else 0.0)
            lead_times.append(max(lead, 0.0))
        ax.bar(x_idx + bi * bar_w, lead_times, bar_w,
                label=f'$L_{{thr}}$={L_thr} voxels',
                color=bar_colors[bi], alpha=0.82,
                edgecolor='k', linewidth=0.9)

    # 在每个工况组中间标一颗星（表示失稳时间参考点 = 0 h lead）
    for i, case in enumerate(cases):
        ax.text(x_idx[i] + bar_w, 0.05,
                f"fail\n@{FAILURE_HOURS.get(case,'?')}h",
                ha='center', va='bottom', fontsize=7.5, color='k')

    ax.set_xticks(x_idx + bar_w)
    ax.set_xticklabels(cases, fontsize=12)
    ax.set_ylabel('Advance warning  $t_{{fail}} - t_{{DAS}}$  (h)', fontsize=11)
    ax.set_title('(f)  Sensitivity to $t_{DAS}$ criterion\n'
                 '(advance warning time vs. $L_{thr}$)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.30, axis='y')

    # ---- 总标题 & 保存 ----
    fig.suptitle(
        'Figure 5 — Main Damage Core Growth:\n'
        'Exponential model fit · Model comparison · Robustness & sensitivity analysis',
        fontsize=13, fontweight='bold', y=1.001,
    )
    out_path = os.path.join(output_dir, 'figure5_lmax_growth.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig5] 已保存: {out_path}")
    return exp_fits, model_fits


# ===========================================================================
#  4. Supplementary Figure S3 — R² / AICc / BIC 热图矩阵
# ===========================================================================

def plot_figure_S3_model_comparison(results, model_fits, output_dir):
    """
    三列热图 (R² / AICc / BIC),行=工况,列=模型。
    右侧附 ΔAICc 差值柱状子图 (相对 Exponential 基准)。
    """
    cases  = sorted(set(r['case'] for r in results))
    models = ['Exponential', 'Linear', 'Power-law', 'Logistic']

    fig = plt.figure(figsize=(17, 5.5))
    # 3 热图 + 1 ΔAICc 柱状
    outer = fig.add_gridspec(1, 4, wspace=0.45, width_ratios=[1, 1, 1, 1.2])

    metric_cfg = [
        ('r2',   'R²',   'RdYlGn',   False, '{:.3f}'),
        ('aicc', 'AICc', 'RdYlGn_r', True,  '{:.1f}'),
        ('bic',  'BIC',  'RdYlGn_r', True,  '{:.1f}'),
    ]

    for col_idx, (met, mlabel, cmap_n, lower_better, fmt) in enumerate(metric_cfg):
        ax  = fig.add_subplot(outer[col_idx])
        mat = np.full((len(cases), len(models)), np.nan)
        for ci, case in enumerate(cases):
            for mj, model in enumerate(models):
                md = model_fits.get(case, {}).get(model)
                if md:
                    mat[ci, mj] = md[met]

        vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0
        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
        im   = ax.imshow(mat, cmap=cmap_n, aspect='auto',
                          vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=28, ha='right', fontsize=9.5)
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(cases, fontsize=11)
        better_str = 'lower = better' if lower_better else 'higher = better'
        ax.set_title(f'{mlabel}   ({better_str})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 数值标注
        rng = max(vmax - vmin, 1e-10)
        for ci in range(len(cases)):
            for mj in range(len(models)):
                v = mat[ci, mj]
                if np.isnan(v):
                    continue
                norm = (v - vmin) / rng
                # 亮色背景 → 黑字；暗色背景 → 白字
                use_white = (norm > 0.60 and not lower_better) or \
                            (norm < 0.40 and lower_better)
                ax.text(mj, ci, fmt.format(v),
                         ha='center', va='center', fontsize=9,
                         fontweight='bold',
                         color='white' if use_white else 'black')

    # ---- ΔAICc 柱状图 (相对 Exponential 基准) ----
    ax_d  = fig.add_subplot(outer[3])
    bar_w = 0.20
    x_idx = np.arange(len(cases))
    comp_models  = ['Linear', 'Power-law', 'Logistic']
    comp_colors  = ['steelblue', 'seagreen', 'darkorange']
    for mi, (mname, mc) in enumerate(zip(comp_models, comp_colors)):
        delta_list = []
        for case in cases:
            mf  = model_fits.get(case, {})
            ref = mf.get('Exponential', {}).get('aicc', np.nan)
            cmp = mf.get(mname,        {}).get('aicc', np.nan)
            delta_list.append(cmp - ref if not (np.isnan(ref) or np.isnan(cmp)) else 0.0)
        ax_d.bar(x_idx + mi * bar_w, delta_list, bar_w,
                  label=mname, color=mc, alpha=0.82,
                  edgecolor='k', linewidth=0.8)

    ax_d.axhline(0, color='k', lw=1.0, ls='--')
    ax_d.axhline(-2, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax_d.axhline(2,  color='gray', lw=0.8, ls=':', alpha=0.7)
    ax_d.text(len(cases)-0.1, 2.3,  'Δ=+2', fontsize=8, color='gray')
    ax_d.text(len(cases)-0.1, -2.5, 'Δ=−2', fontsize=8, color='gray')
    ax_d.set_xticks(x_idx + bar_w)
    ax_d.set_xticklabels(cases, fontsize=12)
    ax_d.set_ylabel('ΔAICc  (model − Exponential)', fontsize=10)
    ax_d.set_title('ΔAICc vs. Exponential\n(positive = worse)',
                   fontsize=11, fontweight='bold')
    ax_d.legend(fontsize=8.5)
    ax_d.grid(alpha=0.30, axis='y')

    fig.suptitle(
        'Supplementary Figure S3 — Model Comparison Matrix\n'
        '(Exponential / Linear / Power-law / Logistic)',
        fontsize=13, fontweight='bold',
    )
    out_path = os.path.join(output_dir, 'figure_S3_model_comparison.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  [S3]  已保存: {out_path}")


# ===========================================================================
#  5. CSV 导出 — Table 3 数据
# ===========================================================================

def export_model_stats_csv(results, exp_fits, model_fits, output_dir):
    """
    导出 Table 3 的完整数据:
        case / model / R² / AICc / BIC
        (Exponential 额外附: k, T_d, k_CI_lo/hi, Td_CI_lo/hi, R²_log10)
    """
    cases  = sorted(set(r['case'] for r in results))
    models = ['Exponential', 'Linear', 'Power-law', 'Logistic']

    rows = []
    for case in cases:
        for model in models:
            md = model_fits.get(case, {}).get(model)
            if md is None:
                continue
            row = dict(case=case, model=model,
                       R2=round(md['r2'], 4),
                       AICc=round(md['aicc'], 2),
                       BIC=round(md['bic'], 2))
            if model == 'Exponential':
                ef = exp_fits.get(case)
                if ef:
                    row.update(
                        k_ln=round(ef['k'], 4),
                        T_d_h=round(ef['T_d'], 3),
                        k_CI95_lo=round(ef['k_ci'][0], 4),
                        k_CI95_hi=round(ef['k_ci'][1], 4),
                        Td_CI95_lo=round(ef['Td_ci'][0], 3)
                                   if not np.isnan(ef['Td_ci'][0]) else np.nan,
                        Td_CI95_hi=round(ef['Td_ci'][1], 3)
                                   if not np.isnan(ef['Td_ci'][1]) else np.nan,
                        R2_log10=round(ef['r2_log'], 4),
                    )
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'table3_model_stats.csv')
    df.to_csv(out_path, index=False)
    print(f"  [CSV] Table 3 已导出: {out_path}")

    # 终端预览
    print("\n  ── Table 3 预览 ──")
    display_cols = ['case', 'model', 'R2', 'AICc', 'BIC', 'k_ln', 'T_d_h',
                    'k_CI95_lo', 'k_CI95_hi']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print()
    return df


# ===========================================================================
#  6. 修改后的 main() — 在原 main() 末尾替换最后几行为以下内容
# ===========================================================================

def main_extended():
    """
    在原 main() 内容基础上追加主损伤核增长分析。
    直接把以下调用加到原 main() 的 export_csv_results(...) 之后即可。
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 保留原有流程 ──
    diagnostics = consistency_check(CSV_FILES)

    if USE_GLOBAL_THRESHOLD:
        print("[第1遍] 构造全局阈值...")
        all_inside = []
        for path, case, label, hours in CSV_FILES:
            if not os.path.exists(path):
                continue
            V, xs, ys, zs = load_voxler_csv(path)
            slope_mask, _, _, _ = build_slope_mask(xs, ys, zs)
            all_inside.append(V[slope_mask])
        all_inside = np.concatenate(all_inside)
        global_thresholds = {p: float(np.percentile(all_inside, p))
                             for p in THRESHOLD_PERCENTILES}
        print(f"  全局阈值: {global_thresholds}\n")
    else:
        global_thresholds = None

    print("[第2遍] 逐时刻分析...")
    results = []
    for path, case, label, hours in CSV_FILES:
        if not os.path.exists(path):
            continue
        print(f"  > {case} {label} ({hours} h)")
        r = analyze_snapshot(path, case, label, hours,
                             global_thresholds=global_thresholds)
        results.append(r)

    # 原有演化图 + CSV
    print("\n[第3遍] 原始演化图...")
    if len(results) >= 2:
        evopath = os.path.join(OUTPUT_DIR, 'evolution_by_case.png')
        plot_evolution_by_case(results, evopath)
    export_csv_results(results, OUTPUT_DIR)

    # ── 新增:主损伤核增长分析 ──
    print("\n[第4遍] 主损伤核增长曲线与稳健性分析...")
    if len(results) >= 3:
        exp_fits, model_fits = plot_figure5_lmax_growth(results, OUTPUT_DIR)
        plot_figure_S3_model_comparison(results, model_fits, OUTPUT_DIR)
        export_model_stats_csv(results, exp_fits, model_fits, OUTPUT_DIR)

    print("\n完成。重点查看:")
    print("  figure5_lmax_growth.png    ← 投稿主图")
    print("  figure_S3_model_comparison.png  ← 补充图 S3")
    print("  table3_model_stats.csv     ← Table 3 数据")


if __name__ == '__main__':
    main_extended()

