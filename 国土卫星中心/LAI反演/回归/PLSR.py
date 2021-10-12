import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_csv('C:/Users/Xizhi Huang/Desktop/Hitters.csv')
print(data.head())

# Salary里有缺失值，直接将所在行删掉
data = data.dropna()

# 将非数字特征变为以0，1替代的特征
dms = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

# 准备数据
y = data['Salary']
x_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
x = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# 训练集、测试集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 回归模型、参数
pls_model_setup = PLSRegression(scale=True,max_iter=5000,n_components=5)
param_grid = {'n_components': range(1, 20)}

# GridSearchCV优化参数、训练模型
gsearch = GridSearchCV(pls_model_setup, param_grid)
pls_model = gsearch.fit(x_train, y_train)

# 打印 coef
print('Partial Least Squares Regression coefficients:', pls_model.best_estimator_.coef_)

# 对测试集做预测
pls_prediction = pls_model.predict(x_test)

# 计算R2，均方差
pls_r2 = r2_score(y_test, pls_prediction)
pls_mse = np.sqrt(mean_squared_error(y_test, pls_prediction))
print(pls_r2)
print(pls_mse)
