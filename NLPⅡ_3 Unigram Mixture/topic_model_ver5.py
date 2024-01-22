#αをトピック別に更新する αのみp.70の式に従う
import numpy as np
from scipy.special import digamma
from scipy import stats
import math
from collections import Counter

class topic_model:
    def __init__(self,K,dataset,dict):
        #datasetの各行（1文書）は語彙文字列（可変長、順番はランダム、重複したものの集約しない）
        #datasetの語彙は、dictにないものが含まれていても大丈夫なように対応できそうだが、
        #バグりそうなので、予めdictにある語彙だけのベクトルに文書をフィルタリングすること（gensimでdictionary
        #のフィルターをやると、出現頻度が低いものや逆に汎用的な語彙を落として辞書を生成するので、もとの
        #文書の形態素ベクトルと一致しなくなる)
        self.no_of_topic = K
        self.dataset=dataset
        self.dict=dict
        self.topic_dst_doc = np.zeros((len(self.dataset),self.no_of_topic))
        self.theta = np.zeros((len(self.dataset),self.no_of_topic))
        self.noOfWords_per_doc = np.array([len(words) for words in self.dataset])
        #辞書型でもlenで要素数を取得できる
        self.word_dst_topic = np.zeros((self.no_of_topic,len(self.dict)))
        self.phi = np.zeros((self.no_of_topic,len(self.dict)))
        self.words_per_topic = np.zeros(self.no_of_topic)
        #zdn(文書dのn番目の語彙のトピックid）に相当　これは、datasetの各行（文書）の語彙列（可変長）と同じ列数なので可変長のリストである
        self.topicId_word_byDoc = [[-1] * len(words) for words in self.dataset]
        self.prob_topicId_byWord = [[0] * len(words) for words in self.dataset]
        #self.topicId_word_byDoc = np.zeros((len(self.dataset),len(self.dict)))
        self.alpha=(np.ones(self.no_of_topic))*0.001
        self.beta=10
        self.sum_alpha=np.sum(self.alpha)
        self.perplecxty_byTopic= np.zeros(self.no_of_topic)
        self.loglikelihood_byTopic= np.zeros(self.no_of_topic)
        
    def lda_gibbs_sampler(self,passes):
        self.loglikelihood = np.zeros(passes)
        self.perplecxty = np.zeros(passes)
        self.perplecxty_byTopic= np.zeros((passes,self.no_of_topic))
        self.loglikelihood_byTopic= np.zeros((passes,self.no_of_topic))
        
        #passesループを繰り返すと、zdnのギブズサンプリングが繰り返される（1回のpassesでは、語彙dnにつきｋ個のギブズサンプリング
        #のみとなる(prob_topicId_word_byDoc[k]の箇所）ので、zdnのサンプルは条件付独立性が十分でない）。
        
        for c in range(passes):
            
            print('alpha',self.alpha)
            print('beta',self.beta)
            #以下のforはギブズサンプリングのためのループ
            for d,words_row in enumerate(self.dataset):
                #print('passes:',c,'doc:',d,'/',len(self.dataset))
                #print(prob_topicId_word_byDoc,'sum=',sum(prob_topicId_word_byDoc))
                for n,word_id in enumerate(words_row):
                #dependencyidをdatasetとする場合は、word_idはsequencenoが入る（０から始まる）
                    #word_id = self.dict[word]
                    #print(word_id)
                    if self.topicId_word_byDoc[d][n] >= 0:#topicId_word_byDocには、初期化状態でない限り
                                                    #0～(K-1)のどれかの番号（トピックID）が入っている
                        self.topic_dst_doc[d][self.topicId_word_byDoc[d][n]]-=1
                        #dictの語彙idがself.word_dst_topicの列番号になっている
                        self.word_dst_topic[self.topicId_word_byDoc[d][n]][word_id]-=1
                        self.words_per_topic[self.topicId_word_byDoc[d][n]]-=1
                    
                    prob_topicId_word_byDoc=np.zeros(self.no_of_topic,dtype = float)
                    topic_dst_doc = np.array(self.topic_dst_doc[d][:])
                    '''
                    print(topic_dst_doc)
                    print(self.alpha)
                    print(topic_dst_doc+self.alpha)
                    print(self.word_dst_topic[:,word_id]+self.beta)
                    print((self.word_dst_topic[:,word_id]+self.beta)/(self.words_per_topic+self.beta*len(self.dict)))
                    input()
                    '''
                    prob_topicId_word_byDoc=((topic_dst_doc+self.alpha)/(self.noOfWords_per_doc[d]+self.sum_alpha))*((self.word_dst_topic[:,word_id]+self.beta)/(self.words_per_topic+self.beta*len(self.dict)))
                    #print(prob_topicId_word_byDoc)
                    '''
                    for k in range(self.no_of_topic):
                        
                        #prob_topicId_word_byDoc[k]=(self.topic_dst_doc[d][k]+self.alpha[k])*(self.word_dst_topic[k][word_id]+self.beta)/(self.words_per_topic[k]+self.beta*len(self.dict))
                        #if math.isnan(prob_topicId_word_byDoc[k]):
                        #   prob_topicId_word_byDoc[k]=1.0
                        prob_topicId_word_byDoc[k]=((self.topic_dst_doc[d][k]+self.alpha[k])/(self.noOfWords_per_doc[d]+self.sum_alpha))*((self.word_dst_topic[k][word_id]+self.beta)/(self.words_per_topic[k]+self.beta*len(self.dict)))
                    ''' 
                    
                    tot=np.sum(prob_topicId_word_byDoc)
                    prob_topicId_word_byDoc=prob_topicId_word_byDoc/tot
                    
                                                                    
                    #print('alpha',self.alpha)
                    #print('beta',self.beta)
                    #print(prob_topicId_word_byDoc,'sum=',sum(prob_topicId_word_byDoc))
                    #probを正規化                   
                    #カテゴリー分布からサンプリング
                    xk = np.arange(self.no_of_topic)                    
                    custm = stats.rv_discrete(name='custm', values=(xk, prob_topicId_word_byDoc))
                    topic = custm.rvs(size=1)
                    self.topicId_word_byDoc[d][n]=topic[0]
                    topic_prob=prob_topicId_word_byDoc[topic[0]]
                    #上記はカテゴリカル分布のサンプリングを1回だけにしたもの。もし、複数回サンプリングして最頻のトピックを算出するなっら以下4行をコメント外す
                    #topics_random = Counter(custm.rvs(size=50))
                    #topic=topics_random.most_common(1)
                    #self.topicId_word_byDoc[d][n]=topic[0][0]
                    #topic_prob=prob_topicId_word_byDoc[topic[0][0]]
                    #print('topic:',topic)
                    
                    self.prob_topicId_byWord[d][n] = topic_prob
                    
                    
                    #self.topicId_word_byDoc[d][n]=np.random.multinomial(1,prob_topicId_word_byDoc)
                    #print(custm.rvs(size=1))
                    #print(self.topicId_word_byDoc[d][n])
                    self.topic_dst_doc[d][self.topicId_word_byDoc[d][n]]+=1
                    #トピック別の語彙分布　列は語彙id nはあくまで文書毎のn番目の単語なので、この単語に対応する
                    #語彙id に変換して以下のv列を更新する
                    #なので、dataset[d][n]=wordの語彙（datasetはid化しない文字でOK)をキーに辞書を参照して語彙id
                    #をインデックスvとする（dictは、辞書型　語彙:id であることが前提）
                    #print(len(self.dict))
                    #print(self.word_dst_topic.shape)
                    self.word_dst_topic[self.topicId_word_byDoc[d][n]][word_id]+=1
                    self.words_per_topic[self.topicId_word_byDoc[d][n]]+=1
            #passesループの最終回になったらα、βの更新をせずに、θ、Φの計算に行く
            
            #for文を1回回したら、α、βを更新。ハイパーパラメータが変わるので、θ、Φも更新される
            #更新したα，βで次のpassesループ（θ、Φは暗黙的に更新される）
            #つまりpasses1回がEMアルゴリズムの1回のループ　E stepはギブズサンプリング、M stepはα、β更新
            num_alpha=np.zeros(self.no_of_topic)
            denom_alpha=np.zeros(self.no_of_topic)
            for dd, words in enumerate(self.dataset):
            
                denom_alpha += digamma(len(words) + self.sum_alpha)
                for kk in range(self.no_of_topic):
                    num_alpha[kk] += digamma(self.topic_dst_doc[dd][kk]+self.alpha[kk])
                        
            denom_alpha=denom_alpha-len(self.dataset)*digamma(self.sum_alpha)
            num_alpha=num_alpha-len(self.dataset)*digamma(self.alpha)
            #denom_alpha=self.no_of_topic*denom_alpha
            self.alpha = self.alpha*num_alpha/denom_alpha
            self.sum_alpha = np.sum(self.alpha)
            
            num_beta=0.0
            denom_beta=0.0
            for kk in range(self.no_of_topic):
                #denom_beta+=digamma(self.words_per_topic[kk] + self.beta*len(self.dict))-self.no_of_topic*digamma(self.beta*len(self.dict))

                denom_beta+=digamma(self.words_per_topic[kk] + self.beta*len(self.dict))
                for v in range(len(self.dict)):
                    #vはword_idと同じである。dependencyidの場合はそのidに付与したsequenceno gensimの語彙辞書の場合は、語彙に付与したsequence no
                    num_beta+=digamma(self.word_dst_topic[kk][v]+self.beta)
                    #print(num_beta)
            denom_beta=denom_beta-self.no_of_topic*digamma(self.beta*len(self.dict))
            denom_beta=len(self.dict)*denom_beta
            num_beta=num_beta-self.no_of_topic*len(self.dict)*digamma(self.beta)
            self.beta = self.beta*num_beta/denom_beta
                
                
        
            #passesループが終了したら、θ、Φを計算する（ギブズサンプリングで十分に条件付独立なzdnを計算した後のパラメータ推定値）
            for dd,topic_dst in enumerate(self.topic_dst_doc):
                for kk,topic_freq in enumerate(topic_dst):
                    self.theta[dd][kk]=(topic_freq+self.alpha[kk])/(self.noOfWords_per_doc[dd]+self.sum_alpha)
            #phiの列番号は、dictの語彙idと一致している
            for kk, word_dst in enumerate(self.word_dst_topic):
                for vv,word_freq in enumerate(word_dst):
                    #vvはword_idと同じである。dependencyidの場合はそのidに付与したsequenceno gensimの語彙辞書の場合は、語彙に付与したsequence no
                    self.phi[kk][vv] = (word_freq + self.beta)/(self.words_per_topic[kk]+self.beta*len(self.dict))
            #返却したいのは、文書別のトピック分布theta, トピック別の語彙確率分布phi（辞書のid順なので辞書から語彙を取得できる）
            #zdn(self.topicId_word_byDoc):各文書d中の語彙nのトピックid datasetと同じ行列形式なので、各zdnがどの語彙
            #に相当するかはdatasetの該当インデックスを参照すればOK。zdnを使うと、元の文書を簡単にトピック列に変換できる
        
            N=0
            for doc, theta in zip(self.dataset,self.theta):
                for w_id in doc:
                    #w_id = self.dict[w]

                    self.loglikelihood[c] -= np.log(np.inner(self.phi[:,w_id], theta))
                    self.loglikelihood_byTopic[c] -= np.log(self.phi[:,w_id]*theta)
                N += len(doc)
            self.perplecxty[c]= np.exp(self.loglikelihood[c] / N)
            self.perplecxty_byTopic[c]= self.loglikelihood_byTopic[c] / N
        
        #return self.theta, self.phi, self.topicId_word_byDoc
        return self.theta, self.phi, self.topicId_word_byDoc,self.prob_topicId_byWord,self.perplecxty,self.perplecxty_byTopic