import pandas as pd
from sklearn.svm import SVC
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

# 3. 初始化和训练SVM模型
# 常用的超参数：
# C: 正则化参数。C越大，模型越能正确分类训练数据，可能会导致过拟合。C越小，模型越平滑，泛化能力更强。
# kernel: 核函数，用于将数据映射到更高维度的空间。常用的核包括 'linear', 'poly', 'rbf', 'sigmoid'。默认是 'rbf'。
# gamma: 核函数的系数。对 'rbf', 'poly', 'sigmoid' 核有用。gamma越大，模型越能拟合训练数据，可能会导致过拟合。可以设置为 'scale'（默认）或 'auto'。
# degree: 多项式核函数的维度（仅在 kernel='poly' 时有效）。
# probability: 是否启用概率估计。设置为 True 时，可以调用 predict_proba 来获得分类概率，但训练时间会增加。
# verbose: 是否输出详细的训练过程信息。verbose=1 会输出训练的进度信息。

model = SVC(C=1.0, kernel='rbf', gamma='scale', degree=3, probability=False, verbose=1)

# 训练模型
model.fit(X_train, y_train)

# 4. 对测试集进行预测
y_pred = model.predict(X_test)

# 5. 计算准确率并输出分类报告
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 6. (可选) 保存模型以便后续使用
joblib.dump(model, 'svm_model.pkl')
