import os
import numpy as np
import cv2
import open3d as o3d

def blur_image(image, blur_type='gaussian', kernel_size=(7, 7), sigma=0):
    """
    对图像进行模糊操作

    参数:
        image (numpy.ndarray): 输入图像
        blur_type (str): 模糊类型，可选值为 'gaussian'（高斯模糊）、'average'（均值模糊）、'median'（中值模糊）
        kernel_size (tuple): 模糊核的大小，对于高斯模糊和均值模糊需要指定
        sigma (int): 高斯模糊的标准差，如果为0则自动计算

    返回:
        numpy.ndarray: 模糊后的图像
    """
    if blur_type == 'gaussian':
        # 高斯模糊
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    elif blur_type == 'average':
        # 均值模糊
        blurred_image = cv2.blur(image, kernel_size)
    elif blur_type == 'median':
        # 中值模糊
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("中值模糊的核大小必须为奇数")
        blurred_image = cv2.medianBlur(image, kernel_size[0])
    else:
        raise ValueError("不支持的模糊类型")

    return blurred_image

def generate_and_display_point_cloud(image_folder, depth_folder, image_name, depth_name):
    # 读取图像和深度图
    image_path = os.path.join(image_folder, image_name)
    depth_path = os.path.join(depth_folder, depth_name)
    
    # 读取图像
    image = cv2.imread(image_path)
    # image = blur_image(image)
    # output_path = r"G:\Experimental_figure\ESWA_v2\endonerf_cutting\mpno_render_7.png"
    # cv2.imwrite(output_path, image)
    # print(f"图像已保存到: {output_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    
    # 读取深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # print(depth.shape)
    
    # 检查图像和深度图的尺寸是否一致
    if image.shape[:2] != depth.shape:
        raise ValueError("图像和深度图的尺寸不一致")
    
    # 归一化深度图到0-1范围
    # depth = depth.astype(np.float32) / 65535.0
    depth = depth.astype(np.float32)
    print(depth.max())
    depth = depth / depth.max()
    # epth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    # 创建Open3D的RGBD图像
    rgb = o3d.geometry.Image(image)
    depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # 转换为毫米单位
    
    # 相机内参（假设使用一个简单的针孔相机模型）
    width, height = image.shape[1], image.shape[0]
    fx, fy = 500.0, 500.0  # 焦距
    cx, cy = width / 2, height / 2  # 图像中心
    
    # 创建相机内参矩阵
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    # 创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    
    return pcd

# 示例用法
# image_folder = r"D:\paper_repro\EndoGaussian\output\endonerf\pulling\video\ours_3000\renders"
# depth_folder = r"D:\paper_repro\EndoGaussian\output\endonerf\pulling\video\ours_3000\depth"
# image_name = "00000.png"  # 替换为实际的图像文件名
# depth_name = "00000.png"  # 替换为实际的深度文件名

image_folder = r"G:\Experimental_figure\ESWA_v2\endonerf_cutting"
depth_folder = r"G:\Experimental_figure\ESWA_v2\endonerf_cutting"
image_name = "mono_render_00090.png"  # 替换为实际的图像文件名
depth_name = "render_depth_00090.png"  # 替换为实际的深度文件名

# generate_and_display_point_cloud(image_folder, depth_folder, image_name, depth_name)

image = cv2.imread(r"G:\Experimental_figure\ESWA_v2\C3VD\gt_00013.png")
image = blur_image(image)
output_path = r"G:\Experimental_figure\ESWA_v2\C3VD\gt_00013_7.png"
cv2.imwrite(output_path, image)
print(f"图像已保存到: {output_path}")