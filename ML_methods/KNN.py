import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
dtype_dict = {i: 'float64' for i in range(8192)}
dtype_dict[8192] = 'int64'  # 最后一列是标签列

# 1. 加载训练集和测试集
train_df = pd.read_csv('tx_train_embedding_2000.csv',header=None,dtype=dtype_dict)
test_df = pd.read_csv('tx_test_embedding.csv',header=None,dtype=dtype_dict)
# 2. 分割特征和标签
X_train = train_df.iloc[:, :-1]  # 取所有行，去掉最后一列作为特征
y_train = train_df.iloc[:, -1]   # 取所有行，最后一列作为标签

X_test = test_df.iloc[:, :-1]    # 测试集特征
y_test = test_df.iloc[:, -1]     # 测试集标签

# 3. 初始化和训练KNN模型
# 常用的超参数：
# n_neighbors: 使用的邻居数量，默认是5。
# weights: 确定每个邻居的权重。'uniform'表示所有邻居权重相等，'distance'表示权重与距离成反比。
# algorithm: 用于计算最近邻的算法。'auto'会根据数据自动选择合适的算法。其他选项包括'ball_tree'、'kd_tree'、'brute'。
# p: 使用的距离度量。p=1表示使用曼哈顿距离，p=2表示使用欧几里得距离。

model = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', p=2)

# 训练模型
model.fit(X_train, y_train)

# 4. 对测试集进行预测
y_pred = model.predict(X_test)
# 检查模型的预测结果
unique_predicted_labels = np.unique(y_pred)
print(f"Unique predicted labels: {unique_predicted_labels}")

# 检查预测结果中是否包含标签 1
if 1 not in unique_predicted_labels:
    print("Model did not predict any instances of label 1.")



# 5. 计算准确率并输出分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred,zero_division=0))

# 6. (可选) 保存模型以便后续使用
joblib.dump(model, 'knn_model.pkl')
