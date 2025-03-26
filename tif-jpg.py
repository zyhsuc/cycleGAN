import os
import rasterio
import numpy as np
from PIL import Image

def convert_tif_to_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)

            # 打开 TIFF 文件
            with rasterio.open(input_path) as src:
                # 读取所有波段
                data = src.read()

                # 检查波段数量
                if data.shape[0] < 4:
                    print(f"文件 {filename} 不是四通道影像，跳过。")
                    continue

                # 提取波段（假设波段顺序为 B, G, R, NIR）
                blue = data[0]  # 蓝色波段
                green = data[1]  # 绿色波段
                red = data[2]  # 红色波段
                nir = data[3]  # 近红外波段

                # 使用 NIR 作为红色通道，R 作为绿色通道，G 作为蓝色通道
                rgb = np.stack([nir, red, green], axis=0)

                # 检查并处理无效值
                rgb = np.nan_to_num(rgb)  # 将 NaN 替换为 0
                rgb[rgb < 0] = 0  # 将负值替换为 0
                rgb[rgb > np.percentile(rgb, 99)] = np.percentile(rgb, 99)  # 去除极端值

                # 归一化数据到 0-255
                rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255
                rgb = rgb.astype(np.uint8).transpose(1, 2, 0)

                # 保存为 JPG
                img = Image.fromarray(rgb)
                img.save(output_path, "JPEG", quality=95)

                print(f"已转换并保存: {output_path}")


# 示例用法
input_folder = r'C:\Users\Lenovo\Desktop\1'  # 替换为你的输入文件夹路径
output_folder = r'C:\Users\Lenovo\Desktop\TEST'  # 替换为你的输出文件夹路径
convert_tif_to_jpg(input_folder, output_folder)