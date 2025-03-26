import rasterio
from rasterio.enums import Resampling
from PIL import Image
import numpy as np
import os



def convert_tif_to_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 TIFF 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

            # 使用 rasterio 打开 TIFF 文件
            with rasterio.open(input_path) as src:
                # 读取所有波段数据
                data = src.read()

                # 检查是否为四通道影像
                if data.shape[0] < 4:
                    print(f"文件 {filename} 不是四通道影像，跳过。")
                    continue

                # 提取 RGB 通道 (假设波段顺序为 B, G, R, NIR)
                # 根据你的数据调整波段索引
                red = data[2]  # R 通道
                green = data[1]  # G 通道
                blue = data[0]  # B 通道

                # 将数据归一化到 [0, 255] 范围
                rgb = np.stack([red, green, blue], axis=0)
                rgb = np.clip(rgb, 0, np.percentile(rgb, 99))  # 去除极端值
                rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255
                rgb = rgb.astype(np.uint8).transpose(1, 2, 0)

                # 使用 Pillow 保存为 JPG
                img = Image.fromarray(rgb)
                img.save(output_path, "PNG", quality=95)

                print(f"已转换并保存: {output_path}")


# 示例用法
input_folder =  r'D:\桌面迁移\科研训练\域适应\域适应\域适应\高分2号_裁切'
output_folder =  r'C:\Users\Lenovo\Desktop\高分'
convert_tif_to_jpg(input_folder, output_folder)