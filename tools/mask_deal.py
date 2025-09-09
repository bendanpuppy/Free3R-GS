import os
import cv2
import numpy as np

def invert_masks(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # 只处理PNG文件
            file_path = os.path.join(folder_path, filename)
            
            # 读取图像
            mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # 检查图像是否成功加载
            if mask is None:
                print(f"无法读取文件: {file_path}")
                continue
            
            # 将图像转换为布尔型
            mask = mask.astype(bool)
            
            # 取反
            inverted_mask = ~mask
            
            # 将布尔型转换回uint8类型（0和255）
            inverted_mask = inverted_mask.astype(np.uint8) * 255
            
            # 保存处理后的图像
            cv2.imwrite(file_path, inverted_mask)
            print(f"已处理并保存: {file_path}")

# 示例用法
folder_path = r"D:\paper_repro\EndoGaussian\data\endonerf\stereo_P2_5_7500_7740\masks"
invert_masks(folder_path)