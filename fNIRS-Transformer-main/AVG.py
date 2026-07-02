import os
import re

# 设置包含 30 个文件夹的根目录路径
root_dir = r"./save/C/LOSO/fNIRS-PreT"  # 替换为实际路径
best_acc_sum = 0
file_count = 0

# 遍历根目录下的每个文件夹
for folder_num in range(1, 31):  # 假设文件夹编号从 1 到 30
    folder_path = os.path.join(root_dir, str(folder_num))
    print(folder_path)
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 构造 test_acc.txt 文件的路径
        file_path = os.path.join(folder_path, "test_acc.txt")
        print(file_path)
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取文件内容
            with open(file_path, "r") as file:
                content = file.read()

                # 使用正则表达式查找 best_acc 的值
                match = re.search(r"best_acc= (\d{1,3}(?:\.\d+)?%?)", content)

                if match:
                    best_acc = match.group(1)
                    # 尝试将 best_acc 转换为浮点数
                    try:
                        best_acc_float = float(best_acc)
                        best_acc_sum += best_acc_float
                        file_count += 1
                        print(f"从文件夹 {folder_num} 中读取到 best_acc = {best_acc_float}")
                    except ValueError:
                        print(f"无法将 best_acc 转换为浮点数: {best_acc}")
                else:
                    print(f"在文件夹 {folder_num} 中未找到 best_acc 的值")
        else:
            print(f"文件夹 {folder_num} 中的 test_acc.txt 文件不存在")
    else:
        print(f"文件夹 {folder_num} 不存在")

# 计算平均值
if file_count > 0:
    average_best_acc = best_acc_sum / file_count
    print(f"\nbest_acc 的平均值为: {average_best_acc:.3f}")
else:
    print("\n未找到任何有效的 best_acc 值")