import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

sc = StandardScaler()

# 读取数据
data = pd.read_csv('C:/Users/Xizhi Huang/Desktop/retestdata.csv')
print(data.head())

# LAI
y = data['LAI']

# 波段
x = data.drop(['LAI'], axis=1)

# 标准化
x_std = sc.fit_transform(x)

# 训练集、测试集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 回归模型、参数
pls_model_setup = PLSRegression(scale=True, max_iter=500, n_components=5)
param_grid = {'n_components': range(1, 20)}

# GridSearchCV优化参数、训练模型
gsearch = GridSearchCV(pls_model_setup, param_grid, cv=5)
pls_model = gsearch.fit(x_train, y_train)

# 打印 coef
print('Partial Least Squares Regression coefficients:', pls_model.best_estimator_.coef_)
a = pls_model.best_estimator_.coef_
# b = pls_model.best_estimator_.intercept_

# 对测试集做预测
pls_prediction = pls_model.predict(x_test)

# 计算R2，均方差
pls_r2 = r2_score(y_test, pls_prediction)
pls_mse = np.sqrt(mean_squared_error(y_test, pls_prediction))
print(pls_r2)
print(pls_mse)

plt.scatter(y_test, pls_prediction)

# axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
# axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,
#                 label='predictions')
# axes[0].set(xlabel='Projected data onto first PCA component',
#             ylabel='y', title='PCR / PCA')
# axes[0].legend()
