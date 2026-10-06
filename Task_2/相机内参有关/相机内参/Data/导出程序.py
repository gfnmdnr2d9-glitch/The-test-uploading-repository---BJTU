import numpy as np
import os

# 改成你的 .npz 文件路径
path = r"C:\Users\***\Documents\***\camera_programs.npz"

# 读取文件
data = np.load(path, allow_pickle=False)

print("包含的数组：", data.files)

for key in data.files:
    arr = data[key]
    print(f"\n{key}: shape={arr.shape}, dtype={arr.dtype}")
    
    # 打印少量数据
    print(arr if arr.size <= 20 else arr.flat[:20])

    # 处理 0 维数组（单个数值）
    if arr.ndim == 0:
        csv_name = f"{key}.csv"
        # 用 [] 把单个数字包起来，变成 1 维数组，就能保存了
        np.savetxt(csv_name, [arr], delimiter=",", fmt="%s")
        print(f"已导出（单值）：{os.path.abspath(csv_name)}")
        
    # 处理 1 维或 2 维数组
    elif arr.ndim <= 2:
        csv_name = f"{key}.csv"
        np.savetxt(csv_name, arr, delimiter=",", fmt="%s")
        print(f"已导出：{os.path.abspath(csv_name)}")
        
    # 其他维度（比如 3D、4D 图像数据）跳过
    else:
        print(f"跳过 {key}，因为它是 {arr.ndim} 维数组，CSV 不支持。")