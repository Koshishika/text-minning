#文書を混合カテゴリカル分布に従うトピック集合とみなして、EM algrithmでカテゴリカル分布のパラメータを推定する
#カテゴリカル分布は以下の2種類からなる
#トピック確率の分布（トピック数no_of_topics分のパラメータthetaをもつ）
#トピック別語彙確率の分布（つまり、トピック数k個分のカテゴリカル分布、各分布は語彙数len(dict)分のパラメータphiを持つ）
#従って、phiはK*len(dinct)次元
#qdkは、文書dがトピックkに属する確率（qdkは、文書docs数分。各文書のqdkはトピック数分の確率値を持つ。
#また、qdkは引数docsの順番となる）
#同時確率doc_phi_jointProbの計算で、語彙数が多いと桁あふれを起こして答えがNaNになるので注意
#(辞書を作成するときにフィルターを使って語彙数を絞る）
#また、乱数の初期値によってもNaNが出るので、何回かトライする

import pandas as pd
import numpy as np
from numpy.random import *
from functools import reduce
from operator import add, mul

#docs:文書（形態素ベクトルを要素とする行列）
#dic:辞書（キーは語彙、値は語彙id)
class em_categorical:

    def __init__(self,docs,dic,no_of_topics):
    
        self.docs=docs # 語彙idの2次元配列
        self.dic=dic #{語彙:id} 
        self.no_of_topics=no_of_topics # クラスタ数
        
    
        #不明なおまじない　なくても動きそう
        eps = 10**(-2)  
    
        #混合比率thetaの初期値を設定（乱数で初期値を設定）  
        self.theta = np.array([random()+eps for k in range(self.no_of_topics)])
        theta_sum=np.sum(self.theta)
        self.theta = self.theta/theta_sum # これは何をやっているか？
    
        # カテゴリカル分布パラメータphi(クラスタの文書特徴を表す語彙確率分布）  phiは何次元か？    
        self.phi = []   
        for k in range(self.no_of_topics):
            self.phi.append([random() for i in range(len(self.dic))])   
        for k in range(self.no_of_topics):
            phi_sum=sum(self.phi[k])        
            self.phi[k]=np.array(self.phi[k])/phi_sum
        
#theta,phiとも一様な値で初期化しても結果は同じ    
#   theta=np.ones(no_of_topics)/no_of_topics
#   phi=rand(no_of_topics,len(dic))
    
    
#   phi_k_total = np.sum(phi, axis=1)
#   print(phi_k_total)
#   for k in range(no_of_topics):
#       phi[k,:]=phi[k,:]/phi_k_total[k]
        
    def fit(self,epoch):
    
        K=self.no_of_topics
        
        qdk=np.zeros(len(self.docs)*K).reshape(len(self.docs),K) # 負担率qdkの初期化
        for e in range(epoch):# E,M stepのイテレーション数
            print('no of step',e+1)
            #theta,phiを更新するための中間的なワーク変数なのでepochのつどゼロクリア
            theta_new=np.zeros(K)
            phi_new=np.zeros(K*len(self.dic)).reshape(K,len(self.dic))
    # -----------------  E-step ------------------------------------------------------  
            for d,doc in enumerate(self.docs):  
                #voc_ids=[w_id for w_id in doc]#     
                #voc_ids=[self.dic[word] for word in doc]#重複した単語をユニーク語彙idリストに変換
                
                doc_phi_jointProb_list=[] # E-step qdkの分子第二項
                for k in range(K):
                
                    #出現語彙のクラスタk別の同時確率（phi[k][id]の積）を求める
                    '''
                    reduce はiterableデータを連続的に演算する（畳み込み）関数
                    array = [20, 1, 2, 3, 4, 5]
                    print(reduce(add, array)) # 35
                    print(reduce(sub, array)) # 5
                    print(reduce(mul, array)) # 2400
                    '''
                    doc_phi_jointProb=reduce(mul,[self.phi[k][id] for id in doc])# E-step qdkの分子第二項（なぜスライドの式と一致するか？）
                    #numpy要素の積の関数を使っても同じ結果
                    #doc_phi_jointProb=np.array([phi[k][id] for id in voc_ids]).prod()
                    #
                    doc_phi_jointProb_list.append(doc_phi_jointProb)
            
                for k in range(K):
                    #上記の同時確率を全てのkについて足し算（qdkの分母）
                    denom_qdk=sum([self.theta[kk]*doc_phi_jointProb_list[kk] for kk in range(K)])
                    #qdkを計算
                    qdk[d][k] = self.theta[k]*doc_phi_jointProb_list[k]/denom_qdk
                
                    #以下、M-stepでtheta,phiを更新するための中間値を求める
                    theta_new[k]+=qdk[d][k]
                
                    for voc_id in doc:
                    
                        #voc_id = self.dic[w]
                        phi_new[k][voc_id]+=qdk[d][k]
                    
                        
    # -----------------------   M-step ---------------------------------------------------------                
            #theta,phi更新（上記のfor d, for kで更新したqdkによって更新。theta_new,phi_newが更新したqdkの情報を持っている代理パラメータ
            self.theta = theta_new / theta_new.sum()
        
            for k in range(K):
                self.phi[k]=phi_new[k]/sum(phi_new[k])
            
        return self.theta,self.phi,qdk 
            



            