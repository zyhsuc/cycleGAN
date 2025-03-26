from PIL import Image
import numpy as np
import os
from glob import glob


def apply_gamma_correction(image_path, output_path, min_percent=0.5, max_percent=99.5, gamma=1):
    """
    1. 对影像进行百分比截断（Percent Clip）
    2. 归一化拉伸到 [0, 255] 之间
    3. 应用 Gamma 拉伸
    :param image_path: 输入图像路径
    :param min_percent: 截断的最小百分位数 (默认 0.5%)
    :param max_percent: 截断的最大百分位数 (默认 99.5%)
    :param gamma: Gamma 值 (>1 使图像变暗, <1 使图像变亮)
    :param output_path: 输出文件路径
    """
    # 读取图像并转换为 NumPy 数组
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.float32)
    # 分别计算 R、G、B 三个通道的百分比截断范围
    min_val = np.percentile(img_array, min_percent, axis=(0, 1))  # 计算 0.5% 处的值
    max_val = np.percentile(img_array, max_percent, axis=(0, 1))  # 计算 99.5% 处的值

    print(f"截断范围：R={min_val[0]}-{max_val[0]}, G={min_val[1]}-{max_val[1]}, B={min_val[2]}-{max_val[2]}")

    # **百分比截断 + 归一化**
    img_array = np.clip(img_array, min_val, max_val) 
    img_array = (img_array - min_val) / (max_val - min_val) * 255.0 
    img_array = np.clip(img_array, 0, 255) 

    # **应用 Gamma 拉伸**
    img_array = (img_array / 255.0) ** gamma * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8) 

    img_corrected = Image.fromarray(img_array)
    img_corrected.save(output_path)

    print(f"百分比截断 + Gamma 变换完成，已保存为 {output_path}")

# 读取图像
in_dir = r"C:\Users\Lenovo\Desktop\高分"
out_dir = r"C:\Users\Lenovo\Desktop\data_set\flower_data\flower_photos\高分转"
os.makedirs(out_dir, exist_ok=True)
imgs = glob(os.path.join(in_dir, '*.png'))
for img in imgs:
    apply_gamma_correction(img, os.path.join(out_dir, os.path.basename(img)))

