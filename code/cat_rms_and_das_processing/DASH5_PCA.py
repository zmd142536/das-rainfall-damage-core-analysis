# -*- coding: utf-8 -*-
"""
DAS 批量去噪流水线
流程：去均值 → 去趋势 → 去尖峰(Despike) → 去共模(Common-Mode) → 去随机噪声(频谱门限+简易维纳) → 多频点陷波(Notch) → PCA去噪
最终输出文件仅保留经过全部处理后的结果。
示例用法：
1. 批量处理整个文件夹
   python das_denoise_h5.py --input_dir "" --output_dir "" \
       --chunk_size 20000 --despike --cmm --rand --noise_ref "0,10" --bp "80,1000" --pca
2. 单文件处理
   python das_denoise_h5.py --input "" --output "" \
       --chunk_size 20000 --despike --cmm --rand --noise_ref "0,10" --bp "80,1000" --pca
"""
import os
import h5py
import argparse
import numpy as np
import logging
from scipy.signal import detrend, stft, istft, butter, filtfilt, iirnotch
import matplotlib
# ------------------------- 日志配置 -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("DAS-DENOISE")
# ------------------------- Matplotlib 样式 -------------------------
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.major.size'] = 4
matplotlib.rcParams['ytick.major.size'] = 4
matplotlib.rcParams['axes.grid'] = False
# ------------------------- 工具函数 -------------------------
def nan_clean(x: np.ndarray) -> np.ndarray:
    """将 NaN/Inf 替换为 0，避免计算错误"""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
def bandpass_if_needed(x, fs, bp):
    """可选带通滤波，bp='fmin,fmax'"""
    if bp is None:
        return x
    fmin, fmax = [float(v) for v in bp.split(",")]
    if fmin <= 0 or fmax >= fs/2.0 or fmin >= fmax:
        log.warning("带通参数异常，跳过带通：%s", bp)
        return x
    b, a = butter(4, [fmin/(fs/2), fmax/(fs/2)], btype='bandpass')
    for ch in range(x.shape[1]):
        x[:, ch] = filtfilt(b, a, x[:, ch], method="gust")
    return x
# ------------------------- 去均值 -------------------------
def subtract_mean(chunk: np.ndarray) -> np.ndarray:
    """对每个通道去均值"""
    return chunk - np.mean(chunk, axis=0, keepdims=True)
# ------------------------- 去尖峰 -------------------------
def despike_mad(chunk: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """MAD-鲁棒去尖峰"""
    x = chunk.copy()
    med = np.median(x, axis=0, keepdims=True)
    diff = x - med
    mad = np.median(np.abs(diff), axis=0, keepdims=True)
    mad[mad == 0] = 1e-6
    z = diff / mad
    mask = np.abs(z) > z_thresh
    if np.any(mask):
        for ch in range(x.shape[1]):
            idx = np.where(mask[:, ch])[0]
            if idx.size == 0:
                continue
            sig = x[:, ch]
            for i in idx:
                s = max(0, i-2)
                e = min(x.shape[0], i+3)
                sig[i] = np.median(sig[s:e])
            x[:, ch] = sig
    return x
# ------------------------- 去共模 -------------------------
def common_mode_remove(chunk: np.ndarray, cm_method: str = "median", eps=1e-8, exclude_channels=None) -> np.ndarray:
    """
    去共模：时间点上对通道取中位/均值作为共模，并按相关系数缩放后相减。
    默认排除坏通道：0-77, 499-501
    """
    if exclude_channels is None:
        exclude_channels = list(range(78)) + list(range(497, 502))
    log.info("去共模，排除通道: %s", exclude_channels)
    valid_channels = [i for i in range(chunk.shape[1]) if i not in exclude_channels]
    if not valid_channels:
        log.warning("无有效通道，跳过去共模")
        return chunk
    cm = np.median(chunk[:, valid_channels], axis=1, keepdims=True) if cm_method == "median" \
         else np.mean(chunk[:, valid_channels], axis=1, keepdims=True)
    x = chunk - np.mean(chunk, axis=0, keepdims=True)
    cm0 = cm - np.mean(cm, axis=0, keepdims=True)
    std_x = np.std(x, axis=0, keepdims=True) + eps
    std_c = np.std(cm0, axis=0, keepdims=True) + eps
    cov = (x * cm0).mean(axis=0, keepdims=True)
    corr = np.clip(cov / (std_x * std_c), 0.0, 1.0)
    return chunk - cm @ np.ones((1, chunk.shape[1]), dtype=chunk.dtype) * corr
# ------------------------- 去随机噪声 -------------------------
def spectral_denoise(chunk: np.ndarray, fs: float, noise_ref: np.ndarray = None,
                     nperseg: int = 1024, noverlap: int = 768,
                     gate_db: float = 6.0, floor_db: float = -12.0) -> np.ndarray:
    """STFT + 简易维纳去噪"""
    eps = 1e-12
    nsamp, nch = chunk.shape
    out = np.zeros_like(chunk)
    def estimate_noise_psd(X):
        P = np.abs(X)**2
        return np.quantile(P, 0.1, axis=1)
    for ch in range(nch):
        f, t, Zxx = stft(chunk[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
        Px = np.abs(Zxx)**2
        if noise_ref is not None:
            _, _, Zref = stft(noise_ref[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
            Pn = estimate_noise_psd(Zref)
        else:
            Pn = estimate_noise_psd(Zxx)
        Pn2 = Pn[:, None]
        snr = np.maximum(Px - Pn2, 0.0) / (Pn2 + eps)
        G = snr / (1.0 + snr)
        gate_lin = 10.0 ** (gate_db / 20.0)
        G = np.where(Px < gate_lin * Pn2, G * (Px / (gate_lin * Pn2 + eps)), G)
        floor_lin = 10.0 ** (floor_db / 20.0)
        G = np.clip(G, floor_lin, 1.0)
        Zden = Zxx * G
        _, x_rec = istft(Zden, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
        if x_rec.shape[0] < nsamp:
            x_rec = np.pad(x_rec, (0, nsamp - x_rec.shape[0]))
        out[:, ch] = x_rec[:nsamp]
    return out
# ------------------------- 工频陷波滤波 -------------------------
def apply_notch_filters(data, fs, freqs, q_vals):
    """对每个通道依次应用多频点陷波器"""
    out = data.copy()
    for f0, q in zip(freqs, q_vals):
        b, a = iirnotch(w0=f0, Q=q, fs=fs)
        log.info(f"应用陷波器：{f0} Hz (Q={q})")
        for ch in range(out.shape[1]):
            out[:, ch] = filtfilt(b, a, out[:, ch])
    return out
# ------------------------- PCA去噪 -------------------------
def pca_denoise(data: np.ndarray, variance_ratio: float = 0.95) -> np.ndarray:
    """使用PCA进行去噪，保留解释方差比例为variance_ratio的主成分"""
    # 中心化数据
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    # SVD分解
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    # 计算累计方差解释比例
    explained_var = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(explained_var)
    # 找到满足比例的组件数
    k = np.argmax(cum_var >= variance_ratio) + 1
    log.info(f"PCA保留 {k} 个主成分 (方差比例: {variance_ratio})")
    # 重构数据
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return reconstructed + mean
# ------------------------- 单文件处理 -------------------------
def process_file(input_path: str, output_path: str, chunk_size: int,
                 do_despike: bool, do_cmm: bool, do_rand: bool, do_pca: bool,
                 fs_override: float, noise_ref_win: tuple, bp: str):
    """处理单个 HDF5 文件，并保存最终结果"""
    assert os.path.exists(input_path), f"输入文件不存在：{input_path}"
    # 读取数据
    with h5py.File(input_path, "r") as f:
        dset = f["default"]
        attrs = {k: v for k, v in dset.attrs.items()}
        fs = float(attrs.get("sampling_rate", fs_override if fs_override else 1000.0))
        row_order = attrs.get("row_major_order", "time, channel")
        if isinstance(row_order, bytes):
            row_order = row_order.decode()
        raw = dset[...]
        data = raw.astype(np.float32) if row_order.strip().lower().replace(" ", "") == "time,channel" else raw.T.astype(np.float32)
    n_samples, n_channels = data.shape
    log.info("文件：%s | fs=%.1f Hz | shape=%s", os.path.basename(input_path), fs, data.shape)
    # 可选带通
    if bp is not None:
        data = bandpass_if_needed(data, fs, bp)
    # 噪声参考段
    noise_ref = None
    if do_rand and noise_ref_win is not None:
        t0, t1 = noise_ref_win
        i0, i1 = max(0, int(t0 * fs)), min(n_samples, int(t1 * fs))
        if i1 - i0 >= 2048:
            noise_ref = data[i0:i1].copy()
            log.info("使用噪声参考段：[%ss, %ss]", t0, t1)
    # 分块处理（去均值、去趋势、去尖峰、去共模、去随机噪声）
    out_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = data[start:end, :].copy()
        chunk = nan_clean(subtract_mean(chunk))  # 去均值
        chunk = nan_clean(detrend(chunk, axis=0, type='linear'))  # 去趋势
        if do_despike:
            chunk = nan_clean(despike_mad(chunk, z_thresh=3.5))
        if do_cmm:
            chunk = nan_clean(common_mode_remove(chunk))
        if do_rand:
            chunk = nan_clean(spectral_denoise(chunk, fs, noise_ref=noise_ref))
        out_chunks.append(chunk.astype(np.float32))
    # 拼接
    den = np.concatenate(out_chunks, axis=0)
    # ---------- 工频陷波 (修改版) ----------
    # 原来的 50, 100 留着防身，新增 47.6 和 42.2 (根据频谱图目测)
    # 如果再次跑完还有噪声，可以将 47.6 微调为 47.5 或 47.7
    notch_mains = [46.3, 100]
    
    # 将 Q 值调低！(从 40 或 100 降到 20)
    # 理由：噪声频率可能会抖动 (比如在 47.4 ~ 47.8 之间飘)，
    # Q值太高(切口太窄)容易再次切偏。Q=20 切口较宽，能把飘忽的噪声“一网打尽”。
    notch_mains_q = [20, 30]
    
    log.info("开始应用陷波滤波：%s", notch_mains)
    final_data = apply_notch_filters(den, fs, notch_mains, notch_mains_q)
    # ---------- PCA去噪 ----------
    if do_pca:
        final_data = pca_denoise(final_data, variance_ratio=0.95)
    # 保存最终结果
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("denoised", data=final_data, dtype="float32", compression="gzip")
        hf.attrs["sampling_rate"] = fs
        hf.attrs["row_major_order"] = "time, channel"
        hf.attrs["source_file"] = input_path
    log.info("保存最终结果：%s", output_path)
# ------------------------- 文件夹批量处理 -------------------------
def process_directory(input_dir: str, output_dir: str, chunk_size: int,
                      do_despike: bool, do_cmm: bool, do_rand: bool, do_pca: bool,
                      fs_override: float, noise_ref_win: tuple, bp: str):
    assert os.path.isdir(input_dir), f"输入文件夹不存在：{input_dir}"
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".h5")]
    if not files:
        log.warning("文件夹中未找到任何 .h5 文件")
        return
    log.info("共找到 %d 个 HDF5 文件待处理", len(files))
    for fname in files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "__final.h5")
        try:
            process_file(input_path, output_path, chunk_size, do_despike, do_cmm, do_rand, do_pca,
                         fs_override, noise_ref_win, bp)
        except Exception as e:
            log.error("处理文件失败：%s | 错误信息：%s", input_path, e)
# ------------------------- CLI 参数 -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="批量 DAS 去噪处理（最终结果仅保留 denoised）")
    ap.add_argument("--input", type=str, default=None, help="单个 HDF5 文件路径")
    ap.add_argument("--input_dir", type=str, default=r"", help="输入文件夹路径")
    ap.add_argument("--output", type=str, default=None, help="输出 HDF5 文件路径（仅单文件模式）")
    ap.add_argument("--output_dir", type=str, default=r"", help="输出文件夹路径（批量模式）")
    ap.add_argument("--chunk_size", type=int, default=20000, help="分块大小")
    ap.add_argument("--despike", action="store_true", help="启用去尖峰")
    ap.add_argument("--cmm", action="store_true", help="启用去共模")
    ap.add_argument("--rand", action="store_true", help="启用去随机噪声")
    ap.add_argument("--pca", action="store_true", help="启用PCA去噪")
    ap.add_argument("--noise_ref", type=str, default=None, help="噪声参考段 't0,t1'")
    ap.add_argument("--bp", type=str, default=None, help="可选带通 'fmin,fmax'")
    return ap.parse_args()
# ------------------------- 主入口 -------------------------
if __name__ == "__main__":
    args = parse_args()
    noise_win = tuple(map(float, args.noise_ref.split(","))) if args.noise_ref else None
    # 检查输入
    if args.input and args.input_dir:
        raise ValueError("不能同时指定 --input 和 --input_dir")
    if not (args.input or args.input_dir):
        raise ValueError("必须指定 --input 或 --input_dir")
    # 单文件
    if args.input:
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + "__final.h5"
        process_file(args.input, args.output, args.chunk_size,
                     args.despike, args.cmm, args.rand, args.pca,
                     fs_override=None, noise_ref_win=noise_win, bp=args.bp)
    else:
        # 批量处理文件夹
        if args.output_dir is None:
            args.output_dir = args.input_dir + "_final"
        process_directory(args.input_dir, args.output_dir, args.chunk_size,
                          args.despike, args.cmm, args.rand, args.pca,
                          fs_override=None, noise_ref_win=noise_win, bp=args.bp)

