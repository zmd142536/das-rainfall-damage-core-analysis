# -*- coding: utf-8 -*-
"""
从应变 CSV/Excel 文件中提取指定时间的行，转置为两列（Label + Value）
支持直接复制表格中的时间字符串作为输入
支持 Excel 日期序列号（如 41867 / 32780）作为输入
"""

import pandas as pd


def extract_and_transform(input_file, output_file, target_time, tolerance_seconds=1.0):
    """
    支持标准时间字符串和 Excel 数字序列号的提取工具
    - input_file: .csv / .xlsx / .xls
    - target_time: 时间字符串(如 "2025/07/07 22:38:30") 或 Excel 序列号字符串(如 "41867"、"32780")
    - tolerance_seconds: 容差秒数；若最接近行时间偏差超过该值，将提示是否继续
    """

    # ==================== 1) 读取文件（方案A：按扩展名自动识别） ====================
    suffix = str(input_file).lower()
    if suffix.endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_file)
    elif suffix.endswith(".csv"):
        # 这里用 utf-8-sig，通常更适配 Excel 导出的 UTF-8 CSV
        # 若你的 CSV 实际是 GBK，可改成 encoding='gbk'
        df = pd.read_csv(input_file, sep=",", encoding="utf-8-sig")
    else:
        raise ValueError("不支持的文件格式：仅支持 .csv / .xlsx / .xls")

    if df.shape[1] < 2:
        raise ValueError("文件列数过少：至少需要 2 列（第 1 列为时间列，其余为数据列）")

    time_col = df.columns[0]

    # ==================== 2) 解析时间列：兼容数字序列号 + 字符串混合 ====================
    def parse_flexible_date(series: pd.Series) -> pd.Series:
        """
        更稳健的混合解析：
        - 能转为数字的部分：按 Excel 序列号处理（1899-12-30 基准）
        - 其余部分：按字符串时间解析
        """
        s_numeric = pd.to_numeric(series, errors="coerce")
        result = pd.Series(index=series.index, dtype="datetime64[ns]")

        mask_num = s_numeric.notnull()
        if mask_num.any():
            result.loc[mask_num] = pd.to_datetime(
                s_numeric.loc[mask_num], unit="D", origin="1899-12-30", errors="coerce"
            )

        if (~mask_num).any():
            result.loc[~mask_num] = pd.to_datetime(series.loc[~mask_num], errors="coerce")

        return result

    print(f"正在解析时间列: {time_col} ...")
    df["_parsed_time"] = parse_flexible_date(df[time_col])

    # 若时间列解析失败（全部 NaT），直接报错
    if df["_parsed_time"].isna().all():
        raise ValueError(
            f"时间列 '{time_col}' 解析失败：全部为 NaT。请检查时间列格式是否为可解析的日期/时间或 Excel 序列号。"
        )

    # ==================== 3) 解析目标时间 target_time ====================
    print(f"正在解析目标输入: {target_time}")
    try:
        t_str = str(target_time).strip()
        # 判断是否为数字（允许小数）
        if t_str.replace(".", "", 1).isdigit():
            target_dt = pd.to_datetime(float(t_str), unit="D", origin="1899-12-30")
        else:
            target_dt = pd.to_datetime(t_str)
    except Exception:
        raise ValueError(f"无法解析输入的时间值：'{target_time}'。请检查输入格式。")

    # ==================== 4) 找最接近的行（排除 NaT） ====================
    valid_df = df[df["_parsed_time"].notna()].copy()
    time_diff = (valid_df["_parsed_time"] - target_dt).abs()
    closest_pos = time_diff.idxmin()
    min_diff_seconds = time_diff.loc[closest_pos].total_seconds()

    actual_time_raw = df.loc[closest_pos, time_col]
    actual_time_parsed = df.loc[closest_pos, "_parsed_time"]

    # ==================== 5) 容差检查与交互 ====================
    if min_diff_seconds > float(tolerance_seconds):
        print("\n⚠️ 警告：未找到精确匹配！")
        print(f"   请求时间：{target_dt}")
        print(f"   最接近点：{actual_time_parsed} (原始值: {actual_time_raw})")
        print(f"   时间偏差：{min_diff_seconds:.2f} 秒")
        proceed = input("   是否继续提取该点数据？(y/n): ")
        if proceed.strip().lower() != "y":
            print("操作已取消。")
            return
    else:
        print(f"✅ 找到匹配！时间点：{actual_time_parsed} (偏差: {min_diff_seconds:.4f} 秒)")

    # ==================== 6) 转置为两列并保存 ====================
    target_row = df.drop(columns=["_parsed_time"]).loc[closest_pos]

    result_df = pd.DataFrame(
        {"Label": target_row.index.astype(str), "Value": target_row.values}
    )

    # 写出：用 utf-8-sig，确保 Excel 打开不乱码
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n处理完成！")
    print(f"保存路径：{output_file}")
    print(f"输出行数：{len(result_df)}")


# ==================== 配置区域 ====================

if __name__ == "__main__":
    # 请根据实际情况修改路径
    input_path = r""
    output_path = r""

    # 这里填写你从表格复制出的内容
    # Excel 序列号直接填字符串，如 "41867"、"32780"
    # 标准时间示例："2025/07/07 22:38:30"
    target_val = "36000"

    extract_and_transform(input_path, output_path, target_val, tolerance_seconds=1.0)

