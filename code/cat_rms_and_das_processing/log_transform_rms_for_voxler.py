"""
RMS值转换成LOG，用于显示细小变化
@author: Claude (Modified)
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# 1. 设置文件名（请修改为你实际的文件名）
input_file = r'' 
output_file = r''

try:
    # 2. 读取数据
    # sep=r'\s+' 表示自动处理空格或Tab分隔符
    df = pd.read_csv(input_file, sep=r'\s+')
    
    # 获取第一列的列名（在你给的例子里，列名是 "12480"）
    target_col = df.columns[0]
    
    print(f"正在读取文件... 识别到数据列名为: [{target_col}]")
    print(f"前几行数据预览:\n{df.head(3)}")

    # 3. 进行对数转换
    # 公式：Log_Result = log10(原始值 + 1)
    # 作用：把 0 变成 0，把极大的噪声值压缩，把关键的小幅破坏保留下来
    df['Log_Result'] = np.log10(df[target_col] + 1.0)

    # 4. 保存文件
    # 保存为 TAB 分隔的 TXT，方便 Voxler 识别
    df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')

    print("-" * 30)
    print(f"处理成功！已生成新文件: {output_file}")
    print(f"新文件中包含了一列 'Log_Result'，请在 Voxler 中用这一列作为颜色(Color)映射。")

except Exception as e:
    print(f"发生错误: {e}")

