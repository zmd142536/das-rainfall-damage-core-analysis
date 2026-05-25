

# -*- coding: utf-8 -*-
"""
DAS 批处理脚本：低通(820 Hz) + 降采样(5000 -> 2000 Hz)
方法一优化版：动态长度 + resize，解决 Can't broadcast 报错问题
改进：
1. 输出 HDF5 的时间维度预留冗余 +50 点，防止分块误差导致写入越界。
2. 启用 HDF5 maxshape，支持动态调整维度。
3. 处理完成后调用 resize() 截断到实际写入长度。
4. 增加误差差异检查与警告日志。
输入数据：
- 每文件 600 秒，501 通道，原始采样率 5000 Hz
"""
import argparse
import os
import glob
import time
from typing import Tuple, Optional
import h5py
import numpy as np
from scipy.signal import butter, sosfilt, resample_poly, sosfilt_zi
# -----------------------------
# 实用函数
# -----------------------------
def find_dataset(h5: h5py.File) -> Tuple[h5py.Dataset, str]:
    """自动查找第一个二维数据集"""
    def _dfs(g, path=""):
        for k, v in g.items():
            p = f"{path}/{k}" if path else k
            if isinstance(v, h5py.Dataset) and v.ndim == 2:
                return v, p
            if isinstance(v, h5py.Group):
                r = _dfs(v, p)
                if r: return r
        return None
    r = _dfs(h5)
    if r is None:
        raise RuntimeError("未找到二维数据集，请使用 --dataset-path 指定。")
    return r
def get_dataset(h5: h5py.File, dataset_path: Optional[str]) -> Tuple[h5py.Dataset, str]:
    if dataset_path:
        if dataset_path not in h5:
            raise KeyError(f"数据集路径 {dataset_path} 不存在。")
        dset = h5[dataset_path]
        if dset.ndim != 2:
            raise ValueError(f"{dataset_path} 不是二维数据集。")
        return dset, dataset_path
    return find_dataset(h5)
def detect_time_channel_axes(shape: Tuple[int, int], meta_row_major: str = "time, channel") -> Tuple[int, int]:
    """根据形状与元数据，推断 (time_axis, chan_axis)"""
    if meta_row_major.strip().lower().replace(" ", "") == "time,channel":
        return 0, 1
    if 501 in shape: # 如果某一维是501，则判为通道维
        chan_axis = 0 if shape[0] == 501 else 1
        time_axis = 1 - chan_axis
        return time_axis, chan_axis
    # 回退：假设较大的维度是时间维
    time_axis = 0 if shape[0] >= shape[1] else 1
    chan_axis = 1 - time_axis
    return time_axis, chan_axis
def design_lowpass(fs: float, f_high: float = 820.0, order: int = 4):
    """设计低通滤波器"""
    nyq = fs * 0.5
    if not (0 < f_high < nyq):
        raise ValueError(f"低通截止频率必须在 (0, {nyq}) 内，当前为 {f_high}。")
    sos = butter(order, f_high/nyq, btype="low", output="sos")
    return sos
def create_output_like(h5_in: h5py.File, dset_in: h5py.Dataset, out_path: str,
                       shape_out: Tuple[int, int], time_axis: int, chan_axis: int, dtype=None):
    """
    创建输出 HDF5 文件，支持动态 resize
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h5_out = h5py.File(out_path, "w")
    # 复制顶层属性
    for k, v in h5_in.attrs.items():
        h5_out.attrs[k] = v
    # 输出数据集路径与输入一致
    dset_path = dset_in.name
    parent = h5_out
    if "/" in dset_path:
        parts = dset_path.strip("/").split("/")[:-1]
        for p in parts:
            parent = parent.require_group(p)
    if dtype is None:
        dtype = dset_in.dtype
    # 设置 maxshape 支持 resize
    if time_axis == 0:
        maxshape = (None, shape_out[chan_axis])
    else:
        maxshape = (shape_out[chan_axis], None)
    dset_out = parent.create_dataset(
        name=dset_in.name.split("/")[-1],
        shape=shape_out,
        maxshape=maxshape,
        dtype=dtype,
        chunks=True,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )
    # 复制原数据集属性
    for k, v in dset_in.attrs.items():
        dset_out.attrs[k] = v
    return h5_out, dset_out
def update_metadata_after_downsample(h5_out: h5py.File, fs_in: float, fs_out: float,
                                     n_time_out: int, duration_s: Optional[float]):
    """更新元数据"""
    h5_out.attrs["sampling_rate"] = np.int64(fs_out)
    h5_out.attrs["ns"] = np.int64(n_time_out)
    if duration_s is not None:
        h5_out.attrs["duration"] = np.float64(duration_s)
    hist_line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                 f"lowpass(820 Hz @ {fs_in} Hz) -> downsample(5000->2000 Hz, resample_poly)")
    prev = h5_out.attrs.get("remarks", "")
    try:
        prev = prev.decode("utf-8") if isinstance(prev, (bytes, bytearray)) else prev
    except Exception:
        pass
    new_remarks = (prev + "\n" + hist_line).strip() if prev else hist_line
    h5_out.attrs["remarks"] = new_remarks
# -----------------------------
# 核心处理
# -----------------------------
def process_file(in_path: str, out_path: str, dataset_path: Optional[str],
                 f_high: float = 820.0,
                 fs_in: float = 5000.0, fs_out: float = 2000.0,
                 chunk_len: int = 100_000, channels: Optional[str] = None, iir_order: int = 4):
    """
    单文件处理
    """
    up, down = 2, 5
    scale = fs_out / fs_in
    assert abs(scale - (up / down)) < 1e-9, "当前实现假定 5000->2000 使用 up=2, down=5"
    with h5py.File(in_path, "r") as h5_in:
        dset_in, found_path = get_dataset(h5_in, dataset_path)
        shape_in = dset_in.shape
        # 解析时间和通道轴
        meta_row_major = h5_in.attrs.get("row_major_order", "time, channel")
        time_axis, chan_axis = detect_time_channel_axes(shape_in, meta_row_major)
        # 解析通道选择
        n_chan = shape_in[chan_axis]
        if channels is None:
            chan_idx = np.arange(n_chan)
        else:
            if ":" in channels:
                a, b = channels.split(":")
                start = int(a) if a else 0
                stop = int(b) if b else n_chan
                chan_idx = np.arange(start, stop)
            else:
                chan_idx = np.array([int(x) for x in channels.split(",")])
        n_sel = len(chan_idx)
        # 时间长度
        n_time_in = shape_in[time_axis]
        n_time_out = int(np.floor(n_time_in * scale))
        duration_s = float(h5_in.attrs.get("duration", n_time_in / fs_in))
        # 设计低通滤波器
        sos = design_lowpass(fs=fs_in, f_high=f_high, order=iir_order)
        # 输出 shape，时间维多留 50 点冗余
        shape_out = list(shape_in)
        shape_out[time_axis] = n_time_out + 50
        shape_out[chan_axis] = n_sel
        shape_out = tuple(shape_out)
        h5_out, dset_out = create_output_like(h5_in, dset_in, out_path, shape_out, time_axis, chan_axis, dtype=dset_in.dtype)
        update_metadata_after_downsample(h5_out, fs_in, fs_out, n_time_out, duration_s)
        # 初始化 IIR 状态
        zi_template = sosfilt_zi(sos)
        zi_all = np.tile(zi_template[:, :, None], (1, 1, n_sel))
        # 分块处理
        overlap = 256
        write_pos = 0
        starts = list(range(0, n_time_in, chunk_len - overlap))
        for i, start in enumerate(starts):
            stop = min(start + chunk_len, n_time_in)
            # 读取块数据
            slc = [slice(None), slice(None)]
            slc[time_axis] = slice(start, stop)
            slc[chan_axis] = chan_idx
            x = dset_in[tuple(slc)]
            # 保证形状 (chan, time)
            if chan_axis == 0:
                pass
            else:
                x = x.swapaxes(0, 1)
            # 低通滤波
            y_lp = np.empty_like(x)
            for c in range(n_sel):
                zi_c = zi_all[:, :, c]
                y_c, zf_c = sosfilt(sos, x[c], zi=zi_c * x[c, 0])
                y_lp[c] = y_c
                zi_all[:, :, c] = zf_c
            # 降采样
            y_ds_list = [resample_poly(y_lp[c], up=2, down=5, padtype="line") for c in range(n_sel)]
            y_ds = np.vstack(y_ds_list)
            # 处理 overlap
            overlap_out = int(round(overlap * scale))
            if i > 0:
                y_ds = y_ds[:, overlap_out:]
            # 写入
            n_w = y_ds.shape[1]
            if time_axis == 0:
                dset_out[write_pos:write_pos + n_w, :] = y_ds.swapaxes(0, 1)
            else:
                dset_out[:, write_pos:write_pos + n_w] = y_ds
            write_pos += n_w
        # 处理完成后 resize 到真实写入长度
        if time_axis == 0:
            dset_out.resize((write_pos, n_sel))
        else:
            dset_out.resize((n_sel, write_pos))
        # 长度差异检查
        diff = abs(write_pos - n_time_out)
        if diff > 10:
            print(f"[WARN] {os.path.basename(out_path)} 长度差异较大: 实际 {write_pos}, 目标 {n_time_out}")
        h5_out.close()
# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DAS 批处理：低通(820 Hz) + 降采样(5000->2000 Hz)")
    p.add_argument("--input-dir", default=r"", help="输入 H5 目录")
    p.add_argument("--output-dir", default=r"", help="输出 H5 目录（自动创建）")
    p.add_argument("--pattern", default="*.h5", help="文件通配符，默认 *.h5")
    p.add_argument("--dataset-path", default=None, help="H5 内部数据集路径")
    p.add_argument("--channels", default=None, help="通道选择，例如 '0:501' 或 '245,246,247'")
    p.add_argument("--chunk-len", type=int, default=100_000, help="分块长度，默认 100000")
    p.add_argument("--f-high", type=float, default=500.0, help="低通截止频率 Hz")
    p.add_argument("--iir-order", type=int, default=4, help="IIR Butter 阶数")
    p.add_argument("--fs-in", type=float, default=5000.0, help="输入采样率 Hz")
    p.add_argument("--fs-out", type=float, default=1000.0, help="输出采样率 Hz")
    return p.parse_args()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        print("未找到输入文件")
        return
    for i, in_path in enumerate(paths, 1):
        fname = os.path.basename(in_path)
        out_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}__LP820_DS2000.h5")
        t0 = time.time()
        print(f"[{i}/{len(paths)}] 处理 {fname} -> {os.path.basename(out_path)}")
        try:
            process_file(
                in_path=in_path,
                out_path=out_path,
                dataset_path=args.dataset_path,
                f_high=args.f_high,
                fs_in=args.fs_in,
                fs_out=args.fs_out,
                chunk_len=args.chunk_len,
                channels=args.channels,
                iir_order=args.iir_order,
            )
            dt = time.time() - t0
            print(f"完成：{fname} ({dt:.1f}s)")
        except Exception as e:
            print(f"[ERROR] 处理 {fname} 失败：{e}")
if __name__ == "__main__":
    main()
