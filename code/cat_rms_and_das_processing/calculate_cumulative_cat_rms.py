# -*- coding: utf-8 -*-
"""
DAS 累积 RMS 分析 - 简化版（按整秒输出）

关键改进：
1. 采用**手动设置通道组**
2. 采用**手动设置每个通道组对应的 RMS 阈值**
3. 只在每秒结束时输出，避免快照时机的复杂性
4. 输出时间点都是整数秒（0, 1, 2, 3...）
5. 每个输出点都包含完整的秒数据
6. **【新增】RMS 累积和记录时减去对应阈值，只记录超出阈值的部分**

@author: 13099 (Modified by Gemini, further modified)
"""

import os
import numpy as np
import h5py
from scipy.signal import butter, filtfilt
import pandas as pd
import logging
from glob import glob
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("DAS-Simplified-Manual")

# ================= 配置（已更新）=================
ref_file = r""
target_dir = r"" # <-- 新目标路径
output_dir = r"" # <-- 新输出路径

dataset_name = "/denoised"
# ... 其他参数保持不变 ...

# ===============================================
fs = 1000.0
lowcut = 50.0
highcut = 180.0
order = 2

start_time = 130.0  # 计算背景噪声的起始时间（秒）
end_time = 300.0    # 计算背景噪声的结束时间（秒）
window_size_sec = 0.1 # RMS计算窗口大小（秒）

# --- 关键修改：手动配置通道组和阈值 ---
# 定义通道组：列表的列表，每个子列表包含一组通道的编号（例如：[80, 81, 82]）
channel_groups_manual = [
    # [80, 81, 82],
    # [85, 86, 87],
    # [90, 91, 92],
    # [95, 96, 97],
    # [100, 101, 102],
    # [105, 106, 107],
    # [110, 111, 112],
    # [115, 116, 117],
    # [120, 121, 122],
    # [125, 126, 127],
    # [130, 131, 132],
    # [135, 136, 137],
    # [140, 141, 142],
    # [145, 146, 147],
    # [150, 151, 152],
    # [155, 156, 157],
    # [160, 161, 162],
    # [165, 166, 167],
    # [170, 171, 172],
    # [175, 176, 177],
    
    # [180, 181, 182],
    # [185, 186, 187],
    # [190, 191, 192],
    # [195, 196, 197],
    # [200, 201, 202],
    # [205, 206, 207],
    # [210, 211, 212],
    # [215, 216, 217],
    # [220, 221, 222],
    # [225, 226, 227],
    # [230, 231, 232],
    # [235, 236, 237],
    # [240, 241, 242],
    # [245, 246, 247],
    # [250, 251, 252],
    # [255, 256, 257],
    
    # [262, 263, 264],
    # [267, 268, 269],
    # [272, 273, 274],
    # [277, 278, 279],
    # [282, 283, 284],
    # [287, 288, 289],
    # [292, 293, 294],
    # [297, 298, 299],
    # [302, 303, 304],
    # [307, 308, 309],
    # [312, 313, 314],
    # [317, 318, 319],
    
    # [323],  
    # [324], 
    # [325], 
    # [326], 
    # [327],          # 组1
    # [328, 329, 330,331,332,333,334,335,336,337,338],        # 组2
    # [340], 
    # [341], 
    # [342], 
    # [343],        # 组3
    # [345, 346, 347,348,349,350,351,352,353,354,355],        # 组4
    # [357],  
    # [358],  
    # [359],  
    # [360],        # 组5
    # [362, 363, 364,365,367,368,369,370,371,372],        # 组6
    # [374], 
    # [375],
    # [376],
    # [377],       # 组7
    # [379, 380,381,382,383,384,385,386,387,388,389], 
    # [391],  
    # [392],   
    # [393],       
    # [394], 
    # [396,397,398,399,340,401,402,403,404,405,406],   # 组8
    # [408], 
    # [409], 
    # [410], 
    # [412,413,414,415,416,417,418,419,420,421,422], 
    # [424], 
    # [425], 
    # [426], 
    # [427], 
    # [428,429,430,431,432,433,434,435,436,437,438], 
    # [439], 
    # [440], 
    # [441], 
    # [442], 
    # [443,444,445,446,447,448,449,450,451,452,453], 
    # [455], 
    # [456], 
    # [457], 
    # [458], 
    # [459,460,461,462,463,464,465,466,467,468,469], 
    # [471], 
    # [472], 
    # [473], 
    # [474], 
    # [476,477,478,479,480,481,482,483,484,485,486], 
    # [488], 
    # [489], 
    # [490], 
    # [491], 
    
    
    # [326],
    # [327],
    # [328],
    # [329],
    # [330],
    # [331,332,333,334,335,336,337,338,339,340,341],
    # [343],
    # [344],
    # [345],
    # [346],
    # [350,351,352,353,354,355,356,357,358,359,360],
    # [362],
    # [363],
    # [364],
    # [365],
    # [366,367,368,369,370,371,372,373,374,375,376],
    # [378],
    # [379],
    # [380],
    # [381],
    # [382,383,384,385,386,387,388,389,390,391,392],
    # [394],
    # [395],
    # [396],
    # [397],
    # [398,399,400,401,402,403,404,405,406,407,408],
    # [410],
    # [411],
    # [412],
    # [413,414,415,416,417,418,419,420,421,422,423],
    # [425],
    # [426],
    # [427],
    # [428],
    # [430,431,432,433,434,435,436,437,438,439,440],
    # [442],
    # [443],
    # [444],
    # [445],
    # [446,447,448,449,450,451,452,453,454,455,456],
    # [458],
    # [459],
    # [460],
    # [461],
    # [462,463,464,465,466,467,468,469,470,471,472],
    # [474],
    # [475],
    # [476],
    # [477],
    # [478,479,480,481,482,483,484,485,486,487,488],
    # [490],
    # [491],
    # [492],
    # [493],
    
######S2#########
    [78, 79, 80],
    [83, 84, 85],
    [88, 89, 90],
    [93, 94, 95],
    [98, 99, 100],
    [103, 104, 105],
    [108, 109, 110],
    [113, 114, 115],
    [118, 119, 120],
    [123, 124, 125],
    [128, 129, 130],
    [133, 134, 135],
    [138, 139, 140],
    [143, 144, 145],
    [146, 147, 148],
    [153, 154, 155],
    [158, 159, 160],
    [163, 164, 165],
    [168, 169, 170],
    [173, 174, 175],
    
    [180, 181, 182],
    [185, 186, 187],
    [190, 191, 192],
    [195, 196, 197],
    [200, 201, 202],
    [205, 206, 207],
    [210, 211, 212],
    [215, 216, 217],
    [220, 221, 222],
    [225, 226, 227],
    [230, 231, 232],
    [235, 236, 237],
    [240, 241, 242],
    [245, 246, 247],
    [250, 251, 252],
    [255, 256, 257],
    
    [262, 263, 264],
    [267, 268, 269],
    [272, 273, 274],
    [277, 278, 279],
    [282, 283, 284],
    [287, 288, 289],
    [292, 293, 294],
    [297, 298, 299],
    [302, 303, 304],
    [307, 308, 309],
    [312, 313, 314],
    [317, 318, 319],
    
    [332],  
    [333], 
    [334], 
    [335], 
    [336],          # 组1
    [337, 338, 339,340,341,342,343,344,345,346,347],
    
    [349],  
    [350], 
    [351], 
    [352],          # 组1
    [353, 354, 355,356,357,358,359,360,361,362,363],
    
    [365],  
    [366], 
    [367], 
    [368],          # 组1
    [369, 370, 371,372,373,374,375,376,377,378,379],
    
    [381],  
    [382], 
    [383], 
    [384],          # 组1
    [385, 386, 387,388,389,390,391,392,393,394,395],
    
    [397],  
    [398], 
    [399], 
    [400],          # 组1
    [401, 402, 403,404,405,406,407,408,409,410,411],
    
    [413],  
    [414], 
    [415],           # 组1
    [417, 418, 419,420,421,422,423,424,425,426,427],
    
    [429],  
    [430], 
    [431], 
    [432],          # 组1
    [433, 434, 435,436,437,438,439,440,441,442,443],
    
    [445],  
    [446], 
    [447], 
    [448],         # 组1
    [449, 450, 451,452,453,454,455,456,457,458,459],
    
    [461],  
    [462], 
    [463], 
    [464],           # 组1
    [465, 466, 467,468,469,470,471,472,473,474,475],
     
    [477], 
    [478], 
    [479],
    [480],         # 组1
    [481, 482,483,484,485,486,487,488,489,490,491],
    
    [493],  
    [494], 
    [495], 
    [496], 
    [497],          # 组1
]

# 定义每个通道组对应的 RMS 阈值：与 channel_groups_manual 数量和顺序一致
# 例如：0.015 对应 [80, 81, 82]，0.020 对应 [100, 101, 102]，以此类推
rms_thresholds_manual = [
# 0.00642703,#6倍标准差阈值
# 0.007012,
# 0.0063275,
# 0.00720617,
# 0.00686526,
# 0.00619743,
# 0.00796199,
# 0.00870633,
# 0.00834731,
# 0.00821204,
# 0.00814097,
# 0.00957686,
# 0.00852502,
# 0.00747054,
# 0.00802123,
# 0.00852414,
# 0.00936462,
# 0.01148691,
# 0.01379624,
# 0.00825466,


# 0.00400489,
# 0.00490278,
# 0.00670069,
# 0.00450336,
# 0.00424032,
# 0.00471169,
# 0.00473063,
# 0.0051325,
# 0.00455126,
# 0.00472362,
# 0.00481067,
# 0.00502333,
# 0.00575721,
# 0.00784502,
# 0.00724392,
# 0.00453228,


# 0.0126159,
# 0.01467952,
# 0.01293403,
# 0.0125692,
# 0.01284712,
# 0.01319667,
# 0.01362255,
# 0.01462107,
# 0.01414391,
# 0.01739015,
# 0.01737795,
# 0.0105126,

# 0.00501038,
# 0.00530868,
# 0.00626256,
# 0.00749563,
# 0.00864399,
# 0.01294341,
# 0.0171618,
# 0.01739765,
# 0.01792885,
# 0.01918326,
# 0.0168544,
# 0.00695676,
# 0.00710409,
# 0.00734497,
# 0.00762755,
# 0.0068326,
# 0.01207672,
# 0.01388015,
# 0.0149988,
# 0.01520811,
# 0.00609266,
# 0.00727684,
# 0.00786336,
# 0.0083961,
# 0.0087041,
# 0.00717727,
# 0.02837246,
# 0.03026512,
# 0.0313654,
# 0.01184754,
# 0.00746292,
# 0.00749116,
# 0.00744864,
# 0.00747874,
# 0.00725177,
# 0.00831389,
# 0.00825211,
# 0.00800369,
# 0.00768913,
# 0.00716506,
# 0.00869325,
# 0.00835134,
# 0.00802521,
# 0.00785787,
# 0.00775151,
# 0.01185114,
# 0.01292066,
# 0.01344374,
# 0.01331033,
# 0.00823853,
# 0.00699375,
# 0.00619331,
# 0.00507688,
# 0.0043112,

# ####5倍标准差阈值
# 0.00588221,
# 0.00643867,
# 0.00585004,
# 0.00661852,
# 0.00632953,
# 0.00575989,
# 0.00730128,
# 0.00796452,
# 0.00764785,
# 0.00753231,
# 0.00747561,
# 0.00875688,
# 0.00782831,
# 0.00691353,
# 0.00739501,
# 0.00782597,
# 0.00857697,
# 0.01046072,
# 0.01250824,
# 0.00756172,

# 0.00367265,
# 0.00456804,
# 0.00616403,
# 0.00419706,
# 0.00392203,
# 0.00433923,
# 0.00436438,
# 0.00472165,
# 0.00420263,
# 0.00436343,
# 0.00442063,
# 0.00462231,
# 0.00526735,
# 0.00712078,
# 0.00657856,
# 0.00413323,

# 0.01143113,
# 0.01327743,
# 0.01172206,
# 0.01139927,
# 0.01164982,
# 0.01196754,
# 0.01235801,
# 0.01322862,
# 0.0127932,
# 0.01569093,
# 0.01571928,
# 0.00956228,

# 0.00460994,
# 0.00488557,
# 0.00575142,
# 0.00687075,
# 0.00792841,
# 0.01176902,
# 0.01519227,
# 0.01537392,
# 0.01582145,
# 0.01690021,
# 0.01493014,
# 0.00644375,
# 0.00658664,
# 0.00680853,
# 0.0070624,
# 0.00633361,
# 0.01090529,
# 0.0124805,
# 0.01345436,
# 0.01362986,
# 0.00563734,
# 0.00668815,
# 0.00720239,
# 0.00766876,
# 0.00793854,
# 0.00659309,
# 0.02503097,
# 0.02666676,
# 0.02760128,
# 0.01058945,
# 0.00683506,
# 0.00685982,
# 0.00682371,
# 0.00685015,
# 0.00664502,
# 0.00759987,
# 0.00754438,
# 0.00732404,
# 0.00704378,
# 0.00656909,
# 0.00791471,
# 0.00762142,
# 0.00734234,
# 0.00720012,
# 0.00709404,
# 0.01070137,
# 0.01163429,
# 0.01209108,
# 0.01197698,
# 0.00755681,
# 0.00636279,
# 0.0056386,
# 0.00464878,
# 0.00395971,


######工频后5倍,50-180hz
# 0.0002367,
# 0.00288347,
# 0.00200648,
# 0.00213247,
# 0.00219413,
# 0.00261722,
# 0.00255849,
# 0.00267856,
# 0.00192531,
# 0.00165572,
# 0.00238325,
# 0.00142826,
# 0.00132861,
# 0.00138742,
# 0.00110874,
# 0.00112979,
# 0.00192343,
# 0.00207241,
# 0.00205531,
# 0.00220561,

# 0.00027703,
# 0.00348886,
# 0.00261593,
# 0.00227596,
# 0.00225936,
# 0.0020786,
# 0.00162612,
# 0.00181959,
# 0.00123452,
# 0.00203146,
# 0.0015686,
# 0.00282647,
# 0.00300873,
# 0.00155349,
# 0.00823029,
# 0.00523803,

# 0.00024477,
# 0.01251372,
# 0.00391459,
# 0.00272575,
# 0.00337406,
# 0.00269391,
# 0.002897,
# 0.0025921,
# 0.00308267,
# 0.00401291,
# 0.00402495,
# 0.00404004,

# 0.00076192,
# 0.00155289,
# 0.00259804,
# 0.00360636,
# 0.00454058,
# 0.00332065,
# 0.01076493,
# 0.01121122,
# 0.0109505,
# 0.01007752,
# 0.00376613,
# 0.00294676,
# 0.0041019,
# 0.00540263,
# 0.00618544,
# 0.00312525,
# 0.00825262,
# 0.00832078,
# 0.00793455,
# 0.00715672,
# 0.00329695,
# 0.01599614,
# 0.01910601,
# 0.02209645,
# 0.02431532,
# 0.00550748,
# 0.02820152,
# 0.01684777,
# 0.00853377,
# 0.01247329,
# 0.00742071,
# 0.00730561,
# 0.00700332,
# 0.00635576,
# 0.00416618,
# 0.00399835,
# 0.00342639,
# 0.00302112,
# 0.00283504,
# 0.0023054,
# 0.00354261,
# 0.00377906,
# 0.00403983,
# 0.00431323,
# 0.0019963,
# 0.00823894,
# 0.00746983,
# 0.00625458,
# 0.00506563,
# 0.00277603,
# 0.00571413,
# 0.00432884,
# 0.00263657,
# 0.00090909,
####S2pca后5倍阈值

# 0.00021714,
# 0.00184345,
# 0.0020996,
# 0.00181904,
# 0.00142558,
# 0.00176459,
# 0.00137838,
# 0.00247803,
# 0.00160214,
# 0.00118549,
# 0.00150215,
# 0.00135612,
# 0.00237499,
# 0.0019202,
# 0.00172411,
# 0.0023679,
# 0.00112607,
# 0.00142929,
# 0.00102414,
# 0.00100349,

# 0.00009982,
# 0.00056648,
# 0.0008195,
# 0.00104976,
# 0.00095848,
# 0.00077581,
# 0.00078653,
# 0.00068809,
# 0.00079074,
# 0.00094292,
# 0.0010369,
# 0.00133595,
# 0.00110857,
# 0.00077144,
# 0.00064793,
# 0.00100437,

# 0.00015331,
# 0.00122619,
# 0.00132806,
# 0.00142273,
# 0.00121273,
# 0.00124328,
# 0.00106781,
# 0.00114086,
# 0.00125019,
# 0.00140952,
# 0.00058362,
# 0.00096383,

# 0.00068665,
# 0.00433032,
# 0.007542,
# 0.00943246,
# 0.00994526,
# 0.0019513,
# 0.00662522,
# 0.00566811,
# 0.00432564,
# 0.00357469,
# 0.00168016,
# 0.0020537,
# 0.00270152,
# 0.00292555,
# 0.00292327,
# 0.00079675,
# 0.00306904,
# 0.00346651,
# 0.00345818,
# 0.00313748,
# 0.00091408,
# 0.00249227,
# 0.00309714,
# 0.00394188,
# 0.00473101,
# 0.00151657,
# 0.0051662,
# 0.00594856,
# 0.00583269,
# 0.00492833,
# 0.00083655,
# 0.00201052,
# 0.00249094,
# 0.00293869,
# 0.0028934,
# 0.00079404,
# 0.00264881,
# 0.00224372,
# 0.00175556,
# 0.00134204,
# 0.00050602,
# 0.00134028,
# 0.00135802,
# 0.00150873,
# 0.00163618,
# 0.00061883,
# 0.00190279,
# 0.00167163,
# 0.00137558,
# 0.00123329,
# 0.00076698,
# 0.0041245,
# 0.00418779,
# 0.00355538,
# 0.0021518,
# 0.00027112,


####S4↓pca后5倍阈值
0.00022076,
0.00152649,
0.00166339,
0.0018185,
0.00185529,
0.00221287,
0.00173614,
0.00215331,
0.00313765,
0.00285142,
0.0032061,
0.00305502,
0.00273567,
0.00265153,
0.00267807,
0.00255211,
0.0019496,
0.00230327,
0.00105255,
0.00124702,

0.00019309,
0.0022934,
0.00235216,
0.00159905,
0.00148765,
0.00228636,
0.00154273,
0.00147827,
0.00147967,
0.00151371,
0.00114811,
0.00206371,
0.00244704,
0.00275517,
0.0023301,
0.00151928,

0.00028594,
0.00284712,
0.00358681,
0.0029948,
0.00325264,
0.00246548,
0.0020165,
0.00152585,
0.00152074,
0.00190596,
0.00152117,
0.00105601,

0.00019368,
0.00182038,
0.00291267,
0.00299968,
0.00231241,
0.00081272,
0.00214862,
0.00194237,
0.00179339,
0.00185632,
0.00065037,
0.00170207,
0.00166216,
0.0014761,
0.00139756,
0.00107196,
0.00454525,
0.00471573,
0.00494999,
0.00511678,
0.0015765,
0.00410603,
0.00564531,
0.00801127,
0.00954299,
0.00168971,
0.00974747,
0.00875468,
0.00673428,
0.00139974,
0.00205454,
0.0025657,
0.00292965,
0.00278829,
0.0019658,
0.00643022,
0.01030938,
0.01599215,
0.02164047,
0.00615026,
0.02031698,
0.02398008,
0.02260889,
0.01835319,
0.00292689,
0.00632515,
0.00593349,
0.00506421,
0.00420987,
0.00100585,
0.00235662,
0.00265112,
0.00245604,
0.00162285,
0.00040738,

]
# ------------------------------------

# 自动确定所有需要的通道范围（用于数据加载）
all_channels = sorted(list(set(ch for group in channel_groups_manual for ch in group)))
data_load_range = [all_channels[0], all_channels[-1]] if all_channels else [0, 0]

file_pattern = "*.h5"


# ================= 基础函数（保持不变） =================
def bandpass_filter(data, fs, lowcut, highcut, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = min(data.shape[1] - 1, 3 * (max(len(a), len(b)) - 1))
    return filtfilt(b, a, data, padlen=padlen)

def calc_noise_variance(ref_block, fs, start_time=None, end_time=None):
    total_points = ref_block.shape[0]
    start_idx = 0 if start_time is None else int(start_time * fs)
    end_idx = total_points if end_time is None else int(end_time * fs)
    ref_segment = ref_block[start_idx:end_idx, :]
    mean_vals = np.mean(ref_segment, axis=0, keepdims=True)
    var = np.mean((ref_segment - mean_vals)**2, axis=0)
    return var

def compute_weights(noise_var):
    w = 1.0 / (noise_var + 1e-12)
    w /= np.sum(w)
    return w

def weighted_combine(block, weights):
    return np.sum(block * weights[None, :], axis=1)

def compute_rms(signal, window_size):
    # 确保至少计算一次 RMS
    num_windows = max(1, (len(signal) - window_size) // window_size + 1)
    rms = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        if end > len(signal):
            break
        rms[i] = np.sqrt(np.mean(signal[start:end]**2))
    return rms

def load_hdf5_data(file_path, dataset_name, channel_range):
    """加载HDF5数据，通道范围为[start_channel, end_channel]"""
    try:
        start_channel, end_channel = channel_range
        # 注意：HDF5加载通常是 [start_channel:end_channel+1] 闭区间
        with h5py.File(file_path, 'r') as f:
            # 自动处理范围：加载所有需要的通道
            data = f[dataset_name][:, start_channel:end_channel+1]
        return data.astype(np.float32)
    except Exception as e:
        log.error(f"加载 {file_path} 失败：{e}")
        return None

# ================= 简化版分析器（已修改）=================
class SimplifiedCumulativeAnalyzer:
    """简化版：按整秒输出
    
    【重要修改】：RMS 记录和累积时减去对应阈值，只记录超出阈值的部分
    """
    
    # 接受阈值列表
    def __init__(self, channel_groups, rms_thresholds, fs, window_size_sec):
        self.channel_groups = channel_groups
        self.num_groups = len(channel_groups)
        self.rms_thresholds = np.array(rms_thresholds) # 改为数组方便索引
        self.fs = fs
        self.window_size_sec = window_size_sec
        self.window_size = int(window_size_sec * fs)
        
        # 验证阈值和分组数量
        if len(self.rms_thresholds) != self.num_groups:
             raise ValueError("手动通道组数量和手动 RMS 阈值数量不一致！")
        
        # 累积RMS（超出阈值的部分）
        self.cumulative_rms = np.zeros(self.num_groups)
        
        # 每秒的瞬时RMS（超出阈值的部分，用于对比和调试）
        self.instantaneous_rms = np.zeros(self.num_groups)
        
        # 当前秒的缓冲区：存储的是 (RMS - 阈值) 的值
        self.second_buffer = [[] for _ in range(self.num_groups)]
        self.current_second = -1
        
        # 存储每秒的结果
        self.results = []  # [{second, instantaneous, cumulative}]
        
    def process_file(self, target_data, ref_weights_dict, file_start_time, data_load_start_channel):
        """处理文件"""
        
        # 实际的窗口数量，最后一个不完整的窗口会被忽略
        num_windows = (target_data.shape[0]) // self.window_size
        if num_windows == 0:
            return
        
        # 计算所有通道组的 RMS
        all_rms = []
        for g_idx, channels in enumerate(self.channel_groups):
            # 计算当前通道组相对于加载数据块的索引
            # data_load_start_channel 是加载数据的起始通道号（如 80）
            # channels[0] 是当前通道组的起始通道号（如 80, 100, 150...）
            # 因此，索引是相对于 data_load_start_channel 的偏移
            channel_indices = [ch - data_load_start_channel for ch in channels]
            
            # 检查索引是否越界
            if any(idx < 0 or idx >= target_data.shape[1] for idx in channel_indices):
                 log.error(f"通道组 {channels} 索引越界，跳过。")
                 all_rms.append(np.zeros(num_windows))
                 continue
                 
            target_block = target_data[:num_windows * self.window_size, :][:, channel_indices]
            
            weights = ref_weights_dict[g_idx]
            combined = weighted_combine(target_block, weights)
            # 使用 reshape/transpose 避免循环，一次性计算所有窗口的 RMS
            # 拆分成 (num_windows, window_size) 的块
            blocks = combined.reshape(num_windows, self.window_size)
            rms = np.sqrt(np.mean(blocks**2, axis=1))
            all_rms.append(rms)
        
        # 按时间窗口处理
        for t_idx in range(num_windows):
            window_time = file_start_time + t_idx * self.window_size_sec
            second = int(window_time)
            
            # 如果进入新的一秒
            if second != self.current_second:
                # 先结算上一秒
                if self.current_second >= 0:
                    self._finalize_second()
                
                # 清空缓冲区并更新当前秒
                self.second_buffer = [[] for _ in range(self.num_groups)]
                self.current_second = second
            
            # 收集 RMS 值（减去阈值后的部分）
            for g_idx in range(self.num_groups):
                current_rms = all_rms[g_idx][t_idx]
                threshold = self.rms_thresholds[g_idx] # 使用当前通道组的独立阈值
                
                if current_rms > threshold:
                    # 【关键修改】：记录的是 RMS - 阈值，而不是原始 RMS
                    excess_rms = current_rms - threshold
                    self.second_buffer[g_idx].append(excess_rms)
    
    def _finalize_second(self):
        """结算一秒的数据"""
        # 计算这一秒的瞬时最大 RMS（超出阈值的部分）
        for g_idx in range(self.num_groups):
            buffer = self.second_buffer[g_idx]
            
            if buffer:
                # 瞬时 RMS 取这一秒内所有超出阈值的 RMS 窗口中的最大值
                # 注意：buffer 中存储的已经是 (RMS - 阈值) 的值
                max_excess = max(buffer)
                self.instantaneous_rms[g_idx] = max_excess
                # 累加到总累积值
                self.cumulative_rms[g_idx] += max_excess
            else:
                # 如果没有超过阈值的 RMS 窗口，则瞬时 RMS 为 0.0
                self.instantaneous_rms[g_idx] = 0.0
        
        # 记录这一秒的结果
        self.results.append({
            'second': self.current_second,
            'instantaneous': self.instantaneous_rms.copy(),
            'cumulative': self.cumulative_rms.copy()
        })
    
    def finalize(self):
        """处理最后一秒"""
        if self.current_second >= 0:
            self._finalize_second()
    
    def get_results(self):
        """获取所有结果"""
        if not self.results:
            return None
        
        seconds = [r['second'] for r in self.results]
        instantaneous_matrix = np.array([r['instantaneous'] for r in self.results])
        cumulative_matrix = np.array([r['cumulative'] for r in self.results])
        
        return {
            'seconds': seconds,
            'instantaneous': instantaneous_matrix,
            'cumulative': cumulative_matrix
        }


# ================= 主处理流程（已修改）=================
def process_simplified_manual(ref_file, target_dir, output_dir, channel_groups,
                              rms_thresholds, fs, lowcut, highcut, order, 
                              start_time, end_time, window_size_sec, 
                              dataset_name, file_pattern, data_load_range):
    """手动配置版处理"""
    
    if not channel_groups:
        log.error("手动通道组列表为空！")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    target_files = sorted(glob(os.path.join(target_dir, file_pattern)))
    if not target_files:
        log.error(f"在 {target_dir} 中未找到文件")
        return
    
    log.info(f"找到 {len(target_files)} 个文件")
    log.info(f"数据加载范围（通道号）：{data_load_range}")
    log.info(f"总共 {len(channel_groups)} 个通道组待分析")
    log.info(f"【注意】本版本记录的是 (RMS - 阈值) 的超出部分")

    # ---------------- 1. 加载和预处理参考数据 ----------------
    ref_data = load_hdf5_data(ref_file, dataset_name, data_load_range)
    if ref_data is None:
        return
    
    log.info("对参考数据进行带通滤波...")
    ref_data = bandpass_filter(ref_data, fs, lowcut, highcut, order)
    
    # ---------------- 2. 计算权重 ----------------
    start_channel_load, _ = data_load_range
    ref_weights_dict = {}
    log.info("计算每个通道组的噪声方差和权重...")
    
    for g_idx, channels in enumerate(channel_groups):
        # 计算当前通道组相对于加载数据块的索引
        channel_indices = [ch - start_channel_load for ch in channels]
        
        # 提取参考数据块
        ref_block = ref_data[:, channel_indices]
        
        # 计算权重
        noise_var = calc_noise_variance(ref_block, fs, start_time, end_time)
        weights = compute_weights(noise_var)
        ref_weights_dict[g_idx] = weights
        # log.info(f"  通道组 {channels}: 权重和 = {np.sum(weights):.4f}")
    
    # ---------------- 3. 初始化分析器 ----------------
    analyzer = SimplifiedCumulativeAnalyzer(channel_groups, rms_thresholds, fs,
                                            window_size_sec)
    
    cumulative_time = 0.0
    
    # ---------------- 4. 处理文件 ----------------
    for file_idx, target_file in enumerate(target_files, 1):
        log.info(f"\n处理文件 {file_idx}/{len(target_files)}: {os.path.basename(target_file)}")
        
        target_data = load_hdf5_data(target_file, dataset_name, data_load_range)
        if target_data is None:
            continue
        
        # 滤波
        target_data = bandpass_filter(target_data, fs, lowcut, highcut, order)
        file_duration = target_data.shape[0] / fs
        
        # 分析
        analyzer.process_file(target_data, ref_weights_dict, cumulative_time, start_channel_load)
        
        cumulative_time += file_duration
        
    # ---------------- 5. 完成处理并保存结果 ----------------
    analyzer.finalize()
    
    results = analyzer.get_results()
    if results:
        save_results_manual(results, channel_groups, data_load_range, output_dir)
    else:
        log.warning("分析结果为空，未生成任何数据。")


def save_results_manual(results, channel_groups, data_load_range, output_dir):
    """保存结果（手动配置版）"""
    
    seconds = results['seconds']
    instantaneous = results['instantaneous']
    cumulative = results['cumulative']
    
    # 生成标签，例如：Ch80-82, Ch100-102
    group_labels = [f"Ch{g[0]}-{g[-1]}" for g in channel_groups]
    
    range_str = f"ch{data_load_range[0]}-{data_load_range[1]}"
    
    # 1. 保存非累积 RMS（超出阈值的部分）
    df_inst = pd.DataFrame(instantaneous, columns=group_labels)
    df_inst.insert(0, 'Time(s)', seconds)
    inst_path = os.path.join(output_dir,
        f"instantaneous_rms_excess_manual_{range_str}.csv")  # 文件名添加 excess 标识
    df_inst.to_csv(inst_path, index=False, float_format='%.7f')
    log.info(f"\n✓ 非累积RMS（超出阈值部分）已保存：{inst_path}")
    
    # 2. 保存累积 RMS（超出阈值的部分）
    df_cum = pd.DataFrame(cumulative, columns=group_labels)
    df_cum.insert(0, 'Time(s)', seconds)
    cum_path = os.path.join(output_dir,
        f"cumulative_rms_excess_manual_{range_str}.csv")  # 文件名添加 excess 标识
    df_cum.to_csv(cum_path, index=False, float_format='%.7f')
    log.info(f"✓ 累积RMS（超出阈值部分）已保存：{cum_path}")
    
    # 3. 验证
    log.info(f"\n{'='*60}")
    log.info(f"【验证】理论累积 vs 实际累积（超出阈值部分）")
    
    theoretical_cumulative = np.cumsum(instantaneous, axis=0)
    final_theoretical = theoretical_cumulative[-1]
    final_actual = cumulative[-1]
    
    log.info(f"最终累积值验证:")
    all_match = True
    for i in range(len(group_labels)):
        theoretical = final_theoretical[i]
        actual = final_actual[i]
        diff = abs(theoretical - actual)
        match = np.isclose(theoretical, actual, rtol=1e-6)
        all_match = all_match and match
        
        log.info(f"  {group_labels[i]}: 理论={theoretical:.7f}, 实际={actual:.7f}, 差异={diff:.7f} {'✓' if match else '❌'}")
        
    if all_match:
        log.info(f"\n✓✓✓ 累积逻辑正确！")
    else:
        log.info(f"\n❌ 累积逻辑有误！")
    
    # 4. 统计信息
    log.info(f"\n非累积RMS统计（超出阈值部分）:")
    log.info(f"  时间范围: {seconds[0]} - {seconds[-1]} 秒")
    if instantaneous.size > 0:
        log.info(f"  数值范围: [{instantaneous.min():.7f}, {instantaneous.max():.7f}]")
        log.info(f"  平均值: {instantaneous.mean():.7f}")
        log.info(f"  非零比例: {(instantaneous > 0).sum() / instantaneous.size * 100:.1f}%")
    
    log.info(f"\n最终累积RMS统计（超出阈值部分）:")
    if final_actual.size > 0:
        log.info(f"  数值范围: [{final_actual.min():.7f}, {final_actual.max():.7f}]")
        log.info(f"  平均值: {final_actual.mean():.7f}")
        log.info(f"  标准差: {final_actual.std():.7f}")
        
    # 5. 示例数据
    log.info(f"\n前10秒的示例数据（第1个通道组: {group_labels[0]})：")
    log.info(f"{'时间(s)':<8} {'瞬时RMS(超出)':<15} {'累积RMS(超出)':<15}")
    log.info(f"{'-'*38}")
    for i in range(min(10, len(seconds))):
        log.info(f"{seconds[i]:<8} {instantaneous[i, 0]:<15.7f} {cumulative[i, 0]:<15.7f}")


def main():
    log.info("="*60)
    log.info("DAS 累积 RMS 分析 - 简化版（手动配置）")
    log.info("【重要】本版本记录的是 (RMS - 阈值) 的超出部分")
    log.info("="*60)
    
    # 检查手动配置
    if len(channel_groups_manual) != len(rms_thresholds_manual):
        log.error("致命错误：手动通道组列表和 RMS 阈值列表长度必须一致！")
        return

    start_time_process = datetime.now()
    
    process_simplified_manual(
        ref_file=ref_file,
        target_dir=target_dir,
        output_dir=output_dir,
        channel_groups=channel_groups_manual,     # 使用手动通道组
        rms_thresholds=rms_thresholds_manual,     # 使用手动阈值
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        start_time=start_time,
        end_time=end_time,
        window_size_sec=window_size_sec,
        dataset_name=dataset_name,
        file_pattern=file_pattern,
        data_load_range=data_load_range           # 自动计算出的数据加载范围
    )
    
    duration = (datetime.now() - start_time_process).total_seconds()
    log.info(f"\n完成！耗时: {duration:.2f} 秒")


if __name__ == "__main__":
    main()
