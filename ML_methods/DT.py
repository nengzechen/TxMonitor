import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

# 3. 初始化和训练决策树模型
# 常用的超参数：
# criterion: 用于衡量分裂质量的标准。常用的有 'gini' (默认) 和 'entropy'。
# max_depth: 决策树的最大深度。防止过拟合。默认是 None，即树会一直生长直到所有叶子节点都是纯的，或直到每个叶子节点包含的样本数少于 min_samples_split。
# min_samples_split: 内部节点再划分所需的最小样本数。默认是 2。
# min_samples_leaf: 叶子节点所需的最小样本数。默认是 1。
# max_features: 寻找最佳分割时考虑的最大特征数。可以是一个整数、浮点数、字符串或 None。默认是 None，即考虑所有特征。
# random_state: 控制随机性。设置这个参数可以使得结果是可重复的。

model = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 4. 对测试集进行预测
y_pred = model.predict(X_test)

# 5. 计算准确率并输出分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 6. (可选) 保存模型以便后续使用
joblib.dump(model, 'decision_tree_model.pkl')
