
#αをトピック別に更新する αのみp.70の式に従う
import numpy as np
from scipy.special import digamma
from scipy import stats
import math
from collections import Counter
import time

class topic_model:
    def __init__(self,K,dataset,dict):
        
        self.no_of_topic = K
        self.dataset=dataset
        self.dict=dict
        self.no_of_doc = len(dataset)
        self.v = len(dict)
        
        def normalize(ndarray, axis):
            return ndarray / ndarray.sum(axis = axis, keepdims = True)

        def normalized_random_array(d0, d1):
            ndarray = np.random.rand(d0, d1)
            return normalize(ndarray, axis = 1)
        
        self.alpha0 = 1
        #(b - a) * np.random.rand() + a
        self.beta0 = 1
        self.alpha = self.alpha0 +  np.random.rand(self.no_of_doc,self.no_of_topic)
        self.beta = self.beta0 +  np.random.rand(self.no_of_topic,self.v)
                
        self.perplecxty_byTopic= np.zeros(self.no_of_topic)
        self.loglikelihood_byTopic= np.zeros(self.no_of_topic)
        self.start = time.time()
    def lda_vb(self,passes):
        
        self.loglikelihood = np.zeros(passes)
        self.perplecxty = np.zeros(passes)
        self.perplecxty_byTopic= np.zeros((passes,self.no_of_topic))
        self.loglikelihood_byTopic= np.zeros((passes,self.no_of_topic))
        #alpha_new = np.ones((self.no_of_doc,self.no_of_topic)) * self.alpha0
        #beta_new =  np.ones((self.no_of_topic,self.v)) * self.beta0        
        for c in range(passes):  
            print('pass : ',c)
            dig_alpha = digamma(self.alpha) - digamma(self.alpha.sum(axis = 1, keepdims = True))
            dig_beta = digamma(self.beta) - digamma(self.beta.sum(axis = 1, keepdims = True))

            alpha_new = np.ones((self.no_of_doc,self.no_of_topic)) * self.alpha0
            beta_new = np.ones((self.no_of_topic,self.v)) * self.beta0
            for (d, N_d) in enumerate(self.dataset):
                #print('passes:',c,'doc:',d,'/',len(self.dataset))
                #w_ids = np.array([self.dict[w] for w in N_d])
                w_ids = np.array(N_d)
                q = np.zeros((self.v,self.no_of_topic))
                v, count = np.unique(w_ids, return_counts = True)
                #print(v,w_ids,N_d)
                q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
                q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

                # 以下、alpha_new,beta_newは、教科書では(6.118)(6.119)変分パラメータΘ、Βに相当する。つまりハイパーパラメータではない。
                # 変分ベイズ学習なので、qの計算をfor文で完了したあとで、別途alpha_new, beta_newを計算しても同じ
                alpha_new[d, :] += count.dot(q[v])#1行K列ベクトルができあがる(6.118)文書d中の語彙毎頻度countに潜在トピック行列(v,topic)の内積をとる  (1,topic)ベクトルが
                                                  #できあがる。各列(topic)は、d中の語彙頻度とqのトピック別語彙頻度との内積結果
                beta_new[:, v] += count * q[v].T#K行v列の行列ができあがる　(6.119) d中の語彙頻度とqの該当語彙要素との単なる要素積(各topic一律に同じ値を掛け算）これで、(6.119)

            self.alpha = alpha_new.copy()
            self.beta = beta_new.copy()
            
            theta_est = np.array([np.random.dirichlet(a) for a in self.alpha])
            phi_est = np.array([np.random.dirichlet(b) for b in self.beta])
            #print('alpha',self.alpha)
            #print('beta',self.beta)
            elapsed_time = time.time() - self.start
            minutes = elapsed_time//60
            sec = elapsed_time - minutes*60
            #print ("elapsed_time:{0}min{1}sec".format(minutes,round(sec, 1)))
            N=0
            for doc, theta in zip(self.dataset,theta_est):
                for w in doc:
                    #w_id = self.dict[w]
                    w_id = w
                    self.loglikelihood[c] -= np.log(np.inner(phi_est[:,w_id], theta))
                    self.loglikelihood_byTopic[c] -= np.log(phi_est[:,w_id]*theta)
                N += len(doc)
            self.perplecxty[c]= np.exp(self.loglikelihood[c] / N)
            self.perplecxty_byTopic[c]= self.loglikelihood_byTopic[c] / N


        return theta_est, phi_est,self.perplecxty,self.perplecxty_byTopic