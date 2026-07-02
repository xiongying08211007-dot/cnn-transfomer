import cv2
import numpy as np
import os
from pathlib import Path

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
        print("查询图像没有提取到特征点。")
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
        print(f"图像 {image_file.name} 匹配数量: {matches}")

        if matches > max_matches:
            max_matches = matches
            best_match_path = image_file

    if max_matches >= threshold:
        return best_match_path
    else:
        print(f"未找到匹配度超过阈值 ({threshold}) 的图像。")
        return None

def visualize_matches(query_image_path, target_image_path):
    """可视化匹配结果"""
    query_image = cv2.imread(query_image_path)
    target_image = cv2.imread(target_image_path)

    query_keypoints, query_descriptors = extract_features(query_image_path)
    target_keypoints, target_descriptors = extract_features(target_image_path)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_descriptors, target_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(query_image, query_keypoints, target_image, target_keypoints, matches[:30], None, flags=2)
    cv2.imshow('Matches', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 查询图像路径
    query_image_path = "dataset/find/17_left.jpg"
    # 数据集目录
    dataset_dir = "dataset/ODIR-5K_Training_Dataset"

    # 查找最相似的图像
    best_match = find_similar_image(query_image_path, dataset_dir, threshold=10)

    if best_match:
        print(f"找到最相似的图像: {best_match.name}")
        visualize_matches(query_image_path, str(best_match))
    else:
        print("未找到相似的图像。")