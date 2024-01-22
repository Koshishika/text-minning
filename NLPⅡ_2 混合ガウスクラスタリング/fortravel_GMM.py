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


#def gaussian(x, mean,cov):
#	return	multivariate_normal.pdf(x, mean=mean, cov=cov,allow_singular=True)

def gaussian(x, mean,cov):
	logn=multivariate_normal.logpdf(x, mean, cov, allow_singular=True)
	return np.exp(logn)




def likelihood(X, mean, cov, pi):
	log_l_sum = 0.0
	for k in range(K):				
		for n in range(len(X)):
			log_l_sum += math.log(pi[k])+multivariate_normal.logpdf(X[n], mean=mean[k], cov=cov[k], allow_singular=True)
	return log_l_sum



if __name__ == "__main__":
			
	
	
	
	csv_input = pd.read_csv('fortravel_token.csv', encoding='ms932', sep=',',skiprows=0)
	###
    
    
    
    
    ####

	

	turn = 0
	
	fig = plt.figure()
	while True:
		
		
		# E-step : 現在のパラメータ(μ_k, ∑_k, π_k）を使って、負担率γ(z_ik)を計算
		# gaussian関数の中身に注意。多変量正規分布の確率密度を計算している
		for n in range(N):# nはスライドではi
			# 分母はkによらないので最初に1回だけ計算
			denominator = 0.0
			#print(cov)
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
		
			
		# 収束判定　前回の対数尤度 - 今回の対数尤度　が閾値以下なら収束と見なして終了
		new_like = likelihood(X, mean, cov, pi)
		diff = new_like - like
		print (turn, diff)
		if abs(diff) < 0.1 or turn >10:
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
    
    ###
	for  in :
		dic = {h:m for m,h in zip(mm,headers)}
		dic_sort=sorted(dic.items(), key=lambda x: x[1],reverse=True)[:20]
		print(dic_sort)
	###	