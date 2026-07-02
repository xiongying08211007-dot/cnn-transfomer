import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

def extract_features(image_path):
    """提取图像的特征点和描述符"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(query_descriptors, target_descriptors):
    """匹配特征点并返回匹配数量"""
    if query_descriptors is None or target_descriptors is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_descriptors, target_descriptors)
    return len(matches)

def find_similar_image(query_image_path, dataset_dir, threshold=10):
    """在数据集中找到与查询图像最相似的图像"""
    query_keypoints, query_descriptors = extract_features(query_image_path)
    if query_descriptors is None:
        print("(f查询图像 {query_image_path} 没有提取到特征点。)")
        return None

    max_matches = 0
    best_match_path = None

    # 遍历数据集中的每张图像
    for image_file in Path(dataset_dir).glob('*'):
        if image_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue

        target_keypoints, target_descriptors = extract_features(str(image_file))
        if target_descriptors is None:
            continue

        matches = match_features(query_descriptors, target_descriptors)
        #print(f"图像 {image_file.name} 匹配数量: {matches}")

        if matches > max_matches:
            max_matches = matches
            best_match_path = image_file

    if max_matches >= threshold:
        return best_match_path
    else:
        print(f"未找到匹配度超过阈值 ({threshold}) 的图像。")
        return None

def save_results_to_excel(results, output_file):
    """将结果保存到 Excel 表格中"""
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    # 数据集目录
    dataset_dir = "dataset/ODIR-5K_Training_Dataset"
    # 查询图像路径（1_left.jpg 到 500_left.jpg）
    query_images_dir = "dataset/find"
    # 输出 Excel 文件
    output_excel = "matching_results.xlsx"

    results = []

    # 遍历 1_left.jpg 到 500_left.jpg
    for i in range(121, 501):
        query_image_name = f"{i}_left.jpg"
        query_image_path = os.path.join(query_images_dir, query_image_name)

        if not os.path.exists(query_image_path):
            print(f"查询图像 {query_image_name} 不存在，跳过。")
            continue

        print(f"正在处理查询图像: {query_image_name}")
        best_match = find_similar_image(query_image_path, dataset_dir, threshold=10)

        if best_match:
            results.append({
                "Query Image": query_image_name,
                "Matched Image": best_match.name,
                "Matched Path": str(best_match)
            })
        else:
            results.append({
                "Query Image": query_image_name,
                "Matched Image": "No Match",
                "Matched Path": "N/A"
            })

    # 将结果保存到 Excel
    save_results_to_excel(results, output_excel)
    print("保存成功")