import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

#関数に投入するデータを作成
x = y = np.arange(-5, 5,0.1)
X, Y = np.meshgrid(x, y)
#print(X)
#print(Y)

z = np.c_[X.ravel(),Y.ravel()]
#print(z)
#print(z.shape)

#二次元正規分布の確率密度を返す関数
def gaussian(x):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    print(det)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    print(inv)
    print(np.dot(np.dot((x - mu),inv),(x - mu).T))
    return np.exp(-np.diag(np.dot(np.dot((x - mu),inv),(x - mu).T)/2.0)) / (np.sqrt((2 * np.pi) ** n * det))

#2変数の平均値を指定
mu = np.array([0,0])
#2変数の分散共分散行列を指定
sigma = np.array([[1,0.],[0.,1]])

Z = gaussian(z)
#print(Z.shape)
shape = X.shape
Z = Z.reshape(shape)

#二次元正規分布をplot
fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
plt.show()
