"""

sklearn.ensemble.RandomForestRegressor
(
n_estimators=’warn’,	# 迭代次数，即森林中决策树的数量
criterion=’mse’, 	# 分裂节点所用的标准，可选“gini”, “entropy”，默认“gini”
max_depth=None, 	# 树的最大深度
min_samples_split=2, 	# 拆分内部节点所需的最少样本数
min_samples_leaf=1,		# 在叶节点处需要的最小样本数。
min_weight_fraction_leaf=0.0, 	# 在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。
max_features=’auto’, 	# 寻找最佳分割时要考虑的特征数量
max_leaf_nodes=None,	# 最大叶子节点数，整数，默认为None
min_impurity_decrease=0.0,  # 如果分裂指标的减少量大于该值，则进行分裂。
min_impurity_split=None, 	# 决策树生长的最小纯净度。不推荐使用
bootstrap=True, 	# 是否有放回的随机选取样本
oob_score=False, 	# 是否采用袋外样本来评估模型的好坏。建议True
n_jobs=None,	# 并行计算数。默认是None。一般选择-1,根据计算机核数自动选择
random_state=None,	# 控制bootstrap的随机性以及选择样本的随机性。一般数字是一样的，便于调参
verbose=0, 		# 在拟合和预测时控制详细程度。默认是0。
warm_start=False	# 若设为True则可以再次使用训练好的模型并向其中添加更多的基学习器
)

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

sc = StandardScaler()

data = pd.read_csv(r'C:/Users/Xizhi Huang/Desktop/retestdata.csv')

"""
data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
"""

print(data.head())

Class_label = np.unique(data['LAI'])
print(Class_label)

info_data = data.info()
print(info_data)

# LAI
y = data['LAI']

# 特征
x = data.drop(['LAI'], axis=1)

# 标准化
x_std = sc.fit_transform(x)

# 训练集、测试集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

forest = RandomForestRegressor(n_estimators=100000, criterion='mse', random_state=0,
                               n_jobs=-1, oob_score=True, bootstrap=True)
forest.fit(x_train, y_train)
param_grid = {"max_depth":np.arange(1,11,1)}

# GridSearchCV优化参数、训练模型
#gsearch = GridSearchCV(forest, param_grid, cv=5)
#rfr_model = gsearch.fit(x_train, y_train)

# 下面对训练好的随机森林，完成重要性评估
# feature_importances_  可以调取关于特征重要程度
importances = forest.feature_importances_
print("重要性：", importances)

x_columns = data.columns[1:]
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
    # 到根，根部重要程度高于叶子。
    print("%2d) %-*s %f" % (f + 1, 30, x_columns[indices[f]], importances[indices[f]]))

score_1 = forest.score(x_test, y_test)
print(score_1)
