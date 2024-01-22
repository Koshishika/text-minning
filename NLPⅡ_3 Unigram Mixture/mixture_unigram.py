
import pandas as pd
import numpy as np
import em_algorithm
from sklearn.decomposition import PCA
import codecs
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
import seaborn as sns
matplotlib.pyplot.rcParams['figure.figsize'] = (16.0, 10.0)

#### 混合カテゴリカル分布クラスの実行


with open('dict_inv.pickle', 'rb') as f:# {語彙:id} のような辞書
    dict_inv = pickle.load(f)
with open('dataset.pickle', 'rb') as f:# 口コミ毎の出現語彙idのリスト（2次元配列） bow形式ではない
    all_reviews = pickle.load(f)

word_dic = {i:w for w, i in dict_inv.items()}# {id:語彙} のような辞書
no_of_topics = 5
em_cat=em_algorithm.em_categorical(all_reviews,dict_inv,no_of_topics)# EM algorithmのオブジェクトを生成
passes =10
theta,phi,qdk =em_cat.fit(passes) # E-step, M-stepを実行して、パラメータ推定結果を返す　passes=はE-step, M-stepのイテレーション数


# ===== クラスタ𝜙_𝑘の特徴を表示する　 トピック𝜙_𝑘𝑣を確率の降順に並べて書き出すと、そのクラスタらしさの語彙順に表示できる
topic=[]
for row in phi: # 問題：phi の次元数はいくつか？（ヒント：dict_invは、出現語彙の辞書）
    w_dic = {word_dic[i]:np.round(val,2) for i,val in enumerate(row)}
    w_dic_sort = (sorted(w_dic.items(), reverse=True, key=lambda x:x[1]))
    print(w_dic_sort[:20])
    topic.append(w_dic_sort)
topic_df = pd.DataFrame(topic)
with codecs.open("fortravel_unigram_topics.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    topic_df.to_csv(f, index=False, encoding="ms932", mode='w')

# =====  負担率𝑞_𝑑𝑘 が最大値となるクラスタk を口コミd毎に求める。np.argmax(pdk,axis=1)は何をやっているか？
 
clusters = np.argmax(qdk,axis =1)

# =====  上記のクラスタを、口コミデータにラベル付けする　

fortravel = pd.read_csv('origin_data_filter.csv', encoding='ms932', sep=',',skiprows=0)
assert len(clusters)==len(fortravel), 'unmatch cluster and dataset'
fortravel['cluster']=clusters.reshape(-1,1)
with codecs.open("fortravel_cluster.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    fortravel.to_csv(f, index=False, encoding="ms932", mode='w')


#### Unigram Mixtureはここまで。　以下PCA（説明は省略） ##############################################################################




with open('bow.pickle', 'rb') as f:
    fortravel_bow = pickle.load(f)
feature_matrix = fortravel_bow.values
dict_word = fortravel_bow.columns.tolist()
# BOWをPCAする
pca = PCA(n_components=10)
	
# 教師なしなので、fitの引数は特徴量のみになっている（教師ラベルがない）
pca.fit(feature_matrix)
pca_matrix=pca.components_
pca_score_matrix = pca.transform(feature_matrix)
feature_pca_vectors=pca.components_.T


#特徴量行列の主成分得点ベクトル（列方向が主成分）   
pca_score_matrix = pca.transform(feature_matrix)

#主成分ベクトル行列を転置する（行は語彙特徴量　列は第一主成分から。。）
feature_pca_vectors=pca.components_.T
#語彙の主成分の1，2列目（第一主成分、第2主成分）を取り出す
feature_pca_df = pd.DataFrame(feature_pca_vectors[:,0:2])

#dataframeの列に見出し行をつける
feature_pca_df.columns = ['PC1','PC2']
#第一主成分、第二主成分の二乗和を見出し、sq_sumでdataframeに追加
square_df = pd.DataFrame(feature_pca_df['PC1']**2+feature_pca_df['PC2']**2)
square_df.columns = ['sq_sum']
feature_name_df=pd.DataFrame(dict_word)
feature_name_df.columns = ['words']
#語彙、主成分、二乗和をつなげたdataframeを作る
feature_pca_df = pd.concat([feature_name_df,feature_pca_df,square_df], axis=1)
#主成分値の大きい語彙順にソート
feature_pca_df= feature_pca_df.sort_values(by='sq_sum', ascending=False)
j=len(feature_pca_df)
#10行目まででスライス
feature_pca_df=feature_pca_df[0:20]


#口コミベクトルの主成分得点行列の第一主成分、第二主成分を取り出してデータフレームを生成
pca_score_df = pd.DataFrame(pca_score_matrix[:,0:2])
#pca_score_df = pca_score_df.sample(n=500)
#データフレームに見出しを付ける
pca_score_df.columns = ['PC1','PC2']
pca_score_df = pd.concat([pca_score_df,fortravel['cluster']],axis=1)
#2次元主成分空間上に口コミ（クラスタ）ベクトルをプロットする。また、上記で求めた主成分の値が大きい語彙特徴量も同時にプロットする
#ax = pca_score_df.plot(kind='scatter', x='PC2', y='PC1',style=['ro', 'bs'],s=5, alpha=0.2, figsize=(40,10))
current_palette = sns.color_palette(n_colors=5)
sns.scatterplot(x='PC2', y='PC1',  data=pca_score_df,hue='cluster', alpha = 0.7, s=100, palette=current_palette)
#上記と同じ平面上に、主成分値の大きい語彙をプロットする

c=0
for word, pca2,pca1 in zip(feature_pca_df['words'],feature_pca_df['PC2'],feature_pca_df['PC1']):
    #語彙のベクトルのプロットに語彙ラベルをアノテーションする
    plt.arrow(0,0,pca2*4.5,pca1*4.5,width=0.002,head_width=0.01,head_length=0.04,length_includes_head=True,color='blue')
    plt.annotate(word,xy=(pca2*5.5,pca1*5.),size=14,color='blue')
    
    
plt.show()



'''






pc_x=0
pc_y=1
pcx='PC'+str(pc_x+1)
pcy='PC'+str(pc_y+1)
feature_pca_df = pd.DataFrame(feature_pca_vectors[:,pc_x:pc_y+1])
print(feature_pca_df)
#dataframeの列に見出し行をつける
feature_pca_df.columns = [pcx,pcy]
#第一主成分、第二主成分の二乗和を見出し、sq_sumでdataframeに追加
square_df = pd.DataFrame(feature_pca_df[pcx]**2+feature_pca_df[pcy]**2)
square_df.columns = ['sq_sum']
feature_name_df=pd.DataFrame(dict_word)
feature_name_df.columns = ['words']
#語彙、主成分、二乗和をつなげたdataframeを作る
feature_pca_df = pd.concat([feature_name_df,feature_pca_df,square_df], axis=1)
#主成分値の大きい語彙順にソート
feature_pca_df= feature_pca_df.sort_values(by='sq_sum', ascending=False)
j=len(feature_pca_df)
#10行目まででスライス
feature_pca_df=feature_pca_df[0:10]
print(feature_pca_df)
pca_score_df = pd.DataFrame(pca_score_matrix[:,pc_x:pc_y+1])
pca_score_df.columns = [pcx,pcy]

colors={i:rgb for i,(color, rgb) in enumerate(matplotlib.colors.CSS4_COLORS.items())}
color_dict = {label:colors[i] for i, label in enumerate(np.unique(clusters))}
color_df = pd.DataFrame([color_dict[label] for label in clusters])
color_df.columns = ['color']
pca_score_df=pd.concat([pca_score_df,color_df],axis=1)
	
ax = pca_score_df.plot(kind='scatter', x=pcx, y=pcy,style=['ro', 'bs'],s=10, c=pca_score_df['color'],alpha=1, figsize=(40,10))
#上記と同じ平面上に、主成分値の大きい語彙をプロットする
for clusterId, px,py in zip(clusters,pca_score_df[pcx],pca_score_df[pcy]):
	ax.annotate(clusterId,xy=(px,py),size=6)
	
for word, pca1,pca2 in zip(feature_pca_df['words'],feature_pca_df[pcx],feature_pca_df[pcy]):
#for word, pca1,pca2 in zip(clusters,feature_pca_df[pcx],feature_pca_df[pcy]):

	ax.annotate(word,xy=(pca1*3,pca2*3),size=10,color='blue')
	ax.quiver(0,0,pca1*3,pca2*3,width=0.002,angles='xy',color='blue',scale_units='xy',scale=1)
		
for i,(k,v) in 	enumerate(color_dict.items()):
	ax.text(2.2, 2.8+(i*0.2), k, size=8, color=v)		
plt.show()

print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))	

	#qdk argmaxして　主成分平面上にアノテーションする
'''