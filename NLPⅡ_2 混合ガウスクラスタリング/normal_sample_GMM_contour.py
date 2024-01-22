# 混合ガウス分布のEMアルゴリズム
from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import cm
import math


def scale(X):
    """データ行列Xを属性ごとに標準化したデータを返す"""
    # 属性の数（=列の数）
    col = X.shape[1]
    
    # 属性ごとに平均値と標準偏差を計算
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    # 属性ごとデータを標準化
    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / sigma[i]
    
    return X


def gaussian(x, mean,cov):
    return  multivariate_normal.pdf(x, mean=mean, cov=cov)

def likelihood(X, mean, cov, pi):
    log_l_sum = 0.0
    for k in range(K):              
        for n in range(len(X)):
            log_l_sum += math.log(pi[k])+multivariate_normal.logpdf(X[n], mean=mean[k], cov=cov[k], allow_singular=True)
    return log_l_sum




def contour_plot(mean, cov):
    # xとyの範囲を定義
    x = np.linspace(-2, 3, 20) #等間隔でデータを０から２０まで20個作成
    y = np.linspace(-2, 3, 20) #等間隔でデータを０から２０まで20個作成

    # グリッドポイントを生成
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.flatten(), Y.flatten()]).T

    # 確率密度関数 (PDF) の値を計算
    Z = multivariate_normal.pdf(XY, mean, cov)
    Z = Z.reshape(X.shape)

    # 描画に使うz値を返す
    return X, Y, Z




def plot_sample(X,cluster,mean,cov):
    ax = fig.add_subplot(1,1,1)
    for x,cls in zip(X,cluster):
        ax.scatter(x[0], x[1],color=cm.tab10(cls),alpha=0.5)
        
    for i,(m,var) in enumerate(zip(mean,cov)):
        ax.scatter(m[0], m[1], s=400,marker='*',color=cm.tab10(i))
        x,y,z = contour_plot(m,var)
        #ax.contour(x,y,z,level=4,color=cm.tab10(i))  
        ax.contour(x, y, z, levels=4, colors=cm.tab10(i))

    




if __name__ == "__main__":
            
    
    N_CLUSTERS = 5
    fig = plt.figure()
    
    # 2変量正規混合分布に従うサンプルデータを生成する
    # https://sabopy.com/py/scikit-learn-1/
    dataset = datasets.make_blobs(n_samples=100, centers=N_CLUSTERS, cluster_std=0.5,n_features=2,random_state=0)
    print(dataset)  
    
    labels = dataset[1] #真のクラスタラベル（使わない）
    X = dataset[0] # 観測データ(100サンプル　5つの正規分布から生成）   
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[:, 0], X[:, 1],alpha=0.5)
    plt.pause(0.1)
    input()
    X = scale(X)
    N = len(X)    # データ数
    dim_gauss  = X.shape[1] # データの次元数（2次元）
    
    K = 5  # 混合ガウス分布クラスタの数（外生変数）

    # 平均、分散、混合係率に初期値を与える
    mean = np.random.rand(K,dim_gauss)  # μ_k  2次元のベクトル要素を一様乱数から生成（K個）
    cov = np.zeros((K,dim_gauss,dim_gauss)) # ∑_k　2x2共分散行列を生成（K個の単位行列）
    for k in range(K):
        cov[k] = np.identity(dim_gauss)
    pi = np.random.rand(K) # π_k 混合比率を一様乱数から生成（K個）
    
    # 負担率の空配列を用意
    gamma = np.zeros((N, K))
    
    # 対数尤度の初期値を計算
    like = likelihood(X, mean, cov, pi)

    turn = 0
    
    fig = plt.figure()
    while True:
        
        
        # E-step : 現在のパラメータ(μ_k, ∑_k, π_k）を使って、負担率γ(z_ik)を計算
        # gaussian関数の中身に注意。多変量正規分布の確率密度を計算している
        for n in range(N):# nはスライドではi
            # 分母はkによらないので最初に1回だけ計算
            denominator = 0.0
            for j in range(K):
                denominator += pi[j] * gaussian(X[n], mean[j], cov[j])
            # 各kについて負担率を計算
            for k in range(K):
                gamma[n][k] = pi[k] * gaussian(X[n], mean[k], cov[k]) / denominator #γ(z_ik)
        
        # M-step : 現在の負担率γ(z_ik)を使って、パラメータを再計算 k 毎にループしながらパラメータ(μ_k, ∑_k, π_k）を推定する。
        for k in range(K):
            # Nkを計算する
            Nk = 0.0
            for n in range(N):
                Nk += gamma[n][k]
            
            # 平均μ_kを推定
            mean[k] = np.zeros(dim_gauss)
            for n in range(N):
                mean[k] += gamma[n][k] * X[n]
            mean[k] /= Nk 
            
            # 共分散∑_kを推定
            cov[k] = np.zeros((dim_gauss,dim_gauss))
            for n in range(N):
                temp = X[n] - mean[k]
                cov[k] += gamma[n][k] * temp.reshape(-1, 1) * temp.reshape(1,-1)  # 縦ベクトルx横ベクトル
            cov[k] /= Nk
            
            # 混合比率π_kを推定
            pi[k] = Nk / N
        
        
        cluster = np.argmax(gamma,axis = 1)# 負担率γ(z_ik)が最大のクラスタkを取り出す（各データの所属するクラスタを推定）      
        plot_sample(X,cluster,mean,cov)# 観測データXを、上記で推定した所属クラスタ毎に色分け表示する。また、クラスタ重心μ_kも色分けしてプロットする
        plt.pause(0.0001)
            
        # 収束判定　前回の対数尤度 - 今回の対数尤度　が閾値以下なら収束と見なして終了
        new_like = likelihood(X, mean, cov, pi)
        diff = new_like - like
        print (turn, diff)
        if abs(diff) < 0.1:
            break
        like = new_like
        turn += 1

    # クラスタ重心
    #mean[k, :]
    
    ### mean[k] ★マーク、* X[n]　の座標点を np.argmax(gamma[n][k] = pi[k]) で色分けプロット
    ##cov[k]を等高線で表す
    
    
    # クラスタリング結果
    cluster = np.argmax(gamma,axis = 1)
    print(cluster)