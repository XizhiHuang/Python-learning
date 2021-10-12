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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

sc = StandardScaler()

data = pd.read_csv(r'C:/Users/Xizhi Huang/Desktop/testselect.csv')

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

"""
需要优化的参数：
n_estimators、max_depth、min_samples_leaf、min_samples_split、max_features
"""

# 进行参数优化

# Search optimal hyperparameter

random_state=42
random_forest_seed=np.random.randint(low=1,high=230)

n_estimators_range = [int(x) for x in np.linspace(start=10, stop=20, num=1)]
max_features_range = ['auto', 'sqrt']
max_depth_range = [int(x) for x in np.linspace(10, 50, num=5)]
max_depth_range.append(None)
min_samples_split_range = [int(x) for x in np.linspace(5, 80, num=5)]
min_samples_leaf_range = [int(x) for x in np.linspace(5, 80, num=5)]

rf_hp_range = {'n_estimators': n_estimators_range,
               'max_features': max_features_range,
               'max_depth': max_depth_range,
               'min_samples_split': min_samples_split_range,
               'min_samples_leaf': min_samples_leaf_range
               }
print(rf_hp_range)

rf_model_test_base = RandomForestRegressor()
rf_model_test_random = RandomizedSearchCV(estimator=rf_model_test_base,
                                          param_distributions=rf_hp_range,
                                          n_iter=200,
                                          n_jobs=-1,
                                          cv=3,
                                          verbose=1,
                                          random_state=random_forest_seed
                                          )
rf_model_test_random.fit(x_train, y_train)

best_hp_now = rf_model_test_random.best_params_
print(best_hp_now)

"""
{'n_estimators': 10, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_features': 'auto', 'max_depth': 40}
"""

"""
# Grid Search

random_forest_hp_range_2 = {'n_estimators': [60, 100, 200],
                            'max_features': [12, 13],
                            'max_depth': [350, 400, 450],
                            'min_samples_split': [2, 3]  # Greater than 1
                            # 'min_samples_leaf':[1,2]
                            # 'bootstrap':bootstrap_range
                            }
random_forest_model_test_2_base = RandomForestRegressor()
random_forest_model_test_2_random = GridSearchCV(estimator=random_forest_model_test_2_base,
                                                 param_grid=random_forest_hp_range_2,
                                                 cv=3,
                                                 verbose=1,
                                                 n_jobs=-1)
random_forest_model_test_2_random.fit(train_X, train_Y)

best_hp_now_2 = random_forest_model_test_2_random.best_params_
print(best_hp_now_2)

forest = RandomForestRegressor(n_estimators=200, criterion='mse', random_state=0,
                               n_jobs=-1, oob_score=True, bootstrap=True,
                               min_samples_split=50, max_depth=15, min_samples_leaf=1)
forest.fit(x_train, y_train)
param_grid = {"max_depth": np.arange(1, 11, 1)}

forest_forest = forest.predict(x_test)

a = sklearn.metrics.r2_score(y_test, forest_forest)

print('finish')



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

plt.rcParams['figure.figsize'] = 16, 13
plt.plot([0.3, 2.0], [0.3, 2.0], 'r-')
plt.xlabel('ture_LAI')
plt.ylabel('pre_LAI')
plt.scatter(y_test, forest_forest, alpha=0.1, s=6)
plt.show()

"""


"""

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

"""
