# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 10:03:24 2026
对RMS超阈值累积进行归一化处理，兼容 Excel (.xlsx) 和 CSV 格式
@author: 13099
"""

import pandas as pd
import numpy as np
import os

def normalize_cumulative_rms(input_file, output_file):
    try:
        print(f"正在读取文件: {input_file} ...")
        
        # --- 1. 智能读取 (根据后缀判断是 Excel 还是 CSV) ---
        file_ext = os.path.splitext(input_file)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            # 如果是 Excel 文件
            df = pd.read_excel(input_file)
        else:
            # 如果是 CSV 文件
            # 尝试用 utf-8 读取，如果失败尝试 gbk (防止中文乱码)
            try:
                df = pd.read_csv(input_file)
            except UnicodeDecodeError:
                df = pd.read_csv(input_file, encoding='gbk')
        
        # 检查数据是否至少有两列
        if df.shape[1] < 2:
            print("错误：数据列数不足，无法处理。")
            return

        # --- 2. 分离数据 ---
        # 假设第一列是时间/索引，不参与归一化
        time_col = df.iloc[:, 0]
        data_cols = df.iloc[:, 1:]

        # --- 3. 执行归一化 (Normalization) ---
        print("正在进行归一化计算...")
        # 公式: (X - min) / (max - min)
        df_norm = (data_cols - data_cols.min()) / (data_cols.max() - data_cols.min())
        
        # 处理全0列或常数列导致的除以0 (NaN) 问题
        df_norm = df_norm.fillna(0)

        # --- 4. 重新组合 ---
        result_df = pd.concat([time_col, df_norm], axis=1)

        # --- 5. 智能输出 (根据后缀判断保存为 Excel 还是 CSV) ---
        out_ext = os.path.splitext(output_file)[1].lower()
        
        print(f"正在保存结果到: {output_file} ...")
        if out_ext in ['.xlsx', '.xls']:
            result_df.to_excel(output_file, index=False)
        else:
            result_df.to_csv(output_file, index=False)
            
        print(f"处理成功！")
        
        # 打印前 5 行预览
        print("\n--- 结果预览 (前5行) ---")
        print(result_df.head())

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        print("提示：请检查路径中的文件夹名是否正确，或者尝试将路径中的反斜杠 \\ 改为正斜杠 /")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 建议：在 Windows 路径字符串前加 r，防止转义字符错误
    input_csv = r''
    output_csv = r''
    
    normalize_cumulative_rms(input_csv, output_csv)
