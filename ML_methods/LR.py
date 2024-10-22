import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import Bunch
from tqdm import tqdm
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

# 3. 初始化和训练逻辑回归模型
# 常用的超参数：
# penalty: 正则化方式。常用的有 'l2'（默认）和 'l1'。L2正则化更常用，L1用于特征选择。
# C: 正则化强度的倒数，C越小，正则化强度越大。默认是 1.0。
# solver: 优化算法，用于求解模型参数。常用选项包括:
#   - 'liblinear': 适用于小型数据集，支持L1和L2正则化。
#   - 'saga': 适用于大型数据集，支持L1、L2和弹性网络正则化。
#   - 'lbfgs': 适用于多类分类任务，支持L2正则化。
# max_iter: 最大迭代次数，默认是 100。增加此值可以确保收敛。
# random_state: 控制随机性，设置后可以保证结果的可重复性。
# verbose: 控制训练过程中的输出信息量。verbose=1 会输出基本的训练进度信息。

model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100, random_state=42, verbose=1)

# 训练模型
model.fit(X_train, y_train)

# 4. 对测试集进行预测
y_pred = model.predict(X_test)

# 5. 计算准确率并输出分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 6. (可选) 保存模型以便后续使用
joblib.dump(model, 'logistic_regression_model.pkl')
