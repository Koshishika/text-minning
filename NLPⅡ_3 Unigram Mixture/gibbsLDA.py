#create_dataset_allWords.pyで作成したコーパスデータセットdataset.pickleにもとづきLDAを計算
#1. トピック毎の語彙確率をword_ranking_byTopic0.csvに書き出す（確率の大きい順に語彙を並べる）
#2. コーパスの各語彙に:トピックidをくっつけたレイアウトでagri_news_dataset_topic0.csvに書き出す
#3. 2.と同じレイアウトで、各語彙を原文書の位置(フィルタなし形態素の要素index dataset_indices)に
#	もどしてtopic_marked_sentence0.xlsx, topic_marked_sentence0.csvに書き出す
#   なお、csvの各セルは、トピックid語彙の単位で書き込む。つまり、LDAの対象でなかった形態素語彙は、後方に出現する
#	topicidつき語彙とjoinで１つの文字列に連結されて、１つのセルに書き込まれる。
#4. 各語彙トピックzdnとそのトピック確率をzdn_word_topic_id0.pickle　zdn_proberbility0.pickleに
#	書き込む。これらは入力コーパスの行列の要素数と一致する

import pandas as pd

from collections import defaultdict
import codecs
import topic_model_ver5
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
	
with open('dict_inv.pickle', 'rb') as f2:
    dict_inv = pickle.load(f2)
w_dict={i:w for w,i in dict_inv.items()}	
#dict_inv : 単語：id
#dataset : １文書を形態素のリストにした2次元配列


passes=10
lda = topic_model_ver5.topic_model(5,dataset,dict_inv)
theta,phi,z,z_prob,p,p_byTopic = lda.lda_gibbs_sampler(passes)

	

word_prob_dict={}
word_prob_matrix=[]
for id,word_dist_inTopic in enumerate(phi):
	#確率の小さいものは除いて辞書を参照する
	for i,word_prob in enumerate(word_dist_inTopic):
		#辞書を作成してvalueの降順に並べる
		word_prob_dict[w_dict[i]]=word_prob
	word_prob_list=sorted(word_prob_dict.items(), key=lambda x: -x[1])
	print('topic id:',id)
	print(word_prob_list[0:20])
	#word_prob_matrix.append(word_prob_list)
	word_prob_matrix.append(word_prob_list)

word_prob_df = pd.DataFrame(word_prob_matrix)
with codecs.open("topic_vectors.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    word_prob_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)


print('minimum_perplexity :' , p.min())
print('minimum_preplexity by topic :',p_byTopic[passes-1])
x=np.arange(passes)
plt.plot(x,p)
plt.show()	





