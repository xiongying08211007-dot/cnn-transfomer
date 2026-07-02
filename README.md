# cnn-transfomer
1. 解压后使用 scripts 文件夹中的 mat 文件提取数据集 A、B 和 C。您需要从数据集链接下载 BBCI 库。对于 fNIRS-PreT，需要手动禁用或注释滤波和基线校正代码（在 B_mat2xls.m 和 C_mat2xls.m 中）。

2. 运行 KFold_Train.py 进行 K 折交叉验证训练。运行 KFold_ACC.py 获取结果。

3. 运行 LOSO_Train.py 执行留一被试交叉验证训练。运行 LOSO_Results.py 获取结果。

4. 在训练之前，您需要指定数据集、模型和数据集路径。
