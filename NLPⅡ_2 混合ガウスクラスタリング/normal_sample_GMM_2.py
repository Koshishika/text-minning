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


def gaussian(x, mean, cov):
    pdf_val = multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
    return max(pdf_val, 1e-6)



def likelihood(X, mean, cov, pi):
    log_l_sum = 0.0
    for k in range(K):
        for n in range(len(X)):
            pdf_val = multivariate_normal.pdf(X[n], mean=mean[k], cov=cov[k], allow_singular=True)
            log_l_sum += math.log(pi[k] * max(pdf_val, 1e-6))
    return log_l_sum

def contour_plot(mean, cov):
    # xとyの範囲を定義
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)

    # グリッドポイントを生成
    X, Y = np.meshgrid(x, y)
    XY = np.dstack([X, Y])

    # XYを適切な形状に変形
    pos = np.empty(XY.shape[:-1] + (2,))
    pos[:, :, 0] = XY[:, :, 0]
    pos[:, :, 1] = XY[:, :, 1]

    # 多変量正規分布の確率密度を計算
    Z = multivariate_normal.pdf(pos, mean, cov)

    return X, Y, Z



def plot_sample(X, cluster, mean, cov):
    fig, ax = plt.subplots()
    
    # 各データポイントをクラスタに基づいて色分けしてプロット
    for x, cls in zip(X, cluster):
        ax.scatter(x[0], x[1], color=plt.cm.tab10(cls % 10), alpha=0.5)

    # クラスタの平均（中心）と等高線をプロット
    for i, (m, var) in enumerate(zip(mean, cov)):
        # クラスタの中心をプロット
        ax.scatter(m[0], m[1], s=100, marker='*', color=plt.cm.tab10(i % 10))

        # 等高線用のデータを生成
        x, y, z = contour_plot(m, var)
        
        # クラスタごとに異なる色で等高線を描画
        color = plt.cm.tab10(i % 10)  # カラーマップから色を取得
        ax.contour(x, y, z, levels=4, colors=[color])

    plt.show()

        
    
    




if __name__ == "__main__":
    
    N_CLUSTERS = 5
    fig = plt.figure()
    
    # CSVファイルの読み込み
    csv_input = pd.read_csv('NLPⅡ_2 混合ガウスクラスタリング/fortravel_token.csv', encoding='ms932')

    # 4列目以降のデータを使用
    X = csv_input.iloc[:, 3:].values.astype(np.float64)  # float64 型に変換
    # 無効な値（NaNやinf）を含む行を削除
    X = X[~np.isnan(X).any(axis=1)]
    X = X[~np.isinf(X).any(axis=1)]
    X = scale(X)  # データの標準化

    # データの基本情報
    N = len(X)  # データ数
    dim_gauss = X.shape[1]  # データの次元数
    K = 5  # クラスタの数

    # パラメータの初期化
    max_iter = 100  # 最大反復回数
    threshold = 0.001  # 収束判定の閾値

    np.random.seed(0) 
    mean = np.random.rand(K, dim_gauss).astype(np.float64)  # float64 型を明示的に使用
    epsilon = 1e-6  # 共分散行列が正定値であることを保証するための小さな正の値
    # 共分散行列の初期化
    cov = np.array([np.identity(dim_gauss) * epsilon for _ in range(K)])
    pi = np.random.rand(K)
    pi /= pi.sum()  # πの合計が1になるように調整
    gamma = np.zeros((N, K))

    # 対数尤度の初期値
    like = likelihood(X, mean, cov, pi)
    
    # 共分散行列が正定値であることを保証
    epsilon = 1e-6  # 小さな正の値
    for k in range(K):
        cov[k] += np.eye(dim_gauss) * epsilon

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

        # 各クラスタ毎の重心ベクトルの降順ソート
        headers = csv_input.columns[3:]  # 重心ベクトルの各要素の名前
        for i, m in enumerate(mean):
            dic = {h: val for h, val in zip(headers, m)}
            dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Cluster {i+1} Top Features:")
            print(dic_sort)
            print("\n")
        
        cluster = np.argmax(gamma,axis = 1)# 負担率γ(z_ik)が最大のクラスタkを取り出す（各データの所属するクラスタを推定）      
        plot_sample(X,cluster,mean,cov)# 観測データXを、上記で推定した所属クラスタ毎に色分け表示する。また、クラスタ重心μ_kも色分けしてプロットする
        plt.pause(0.0001)

        # 収束判定
        prev_like = -np.inf
        for _ in range(max_iter):
    # EMアルゴリズムのステップ...
            new_like = likelihood(X, mean, cov, pi)
            if abs(new_like - prev_like) < threshold:
                break
            prev_like = new_like


    # クラスタリング結果の出力
        cluster = np.argmax(gamma, axis=1)
        print(cluster)
