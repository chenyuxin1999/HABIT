from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import sklearn
from demo import *
import numpy as np



j = load_json('Data_1103.json')
data = load_data(j)
fea, label = new_extract_data(data)
feature = func_bin(fea)
label = func_label(label)
x = feature
y = label



# 定义随机森林回归树
reg = RandomForestRegressor(criterion='mse', n_estimators=100, random_state=0)

# 使用交叉验证接口进行测试
scores = cross_val_score(reg, x, y, cv=10, scoring="neg_mean_squared_error")
print(scores)

reg = reg.fit(x, y)

# 使用predict接口，看看预测的效果噻
print(y[0:10])
res = reg.predict(x)
print(res[0:10])
