import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import codecs
import japanize_matplotlib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
matplotlib.pyplot.rcParams['figure.figsize'] = (16.0, 10.0)

csv_input = pd.read_csv('fortravel_bow.csv', encoding='ms932', sep=',',skiprows=0)
#素性をTF-IDFにするなら、create_dataset_tfidfを使う

dict_word=csv_input.columns[3:]
feature_matrix=csv_input.iloc[:,3:].values

vec = KMeans(n_clusters=4)
pred_vec =vec.fit_predict(csv_input.iloc[:,3:])
csv_input["class"]=pred_vec

'''
unique_label = csv_input['keyword'].unique()    
label_dict={}
for label_id,label in enumerate(unique_label):
    
    #辞書に、ラベルテキストをキーとしてvalueにlabel_idを入れる
    label_dict[label] = label_id

label_id_list=[]
#データセットの教師ラベルテキストを整数ラベルに変換
for label_text in csv_input['keyword'].values.tolist():
    
    labelId = label_dict[label_text]        
    label_id_list.append(labelId)       

array_label_id = np.array(label_id_list)
'''   
#主成分数を可変にして、累積寄与率90%になるような主成分数を求める
#PCA()でオブジェクト生成、.fitで特徴量にもとづく学習
pca = PCA(n_components=len(dict_word))

# 教師なしなので、fitの引数は特徴量のみになっている（教師ラベルがない）
pca.fit(feature_matrix)
contrib_list=pca.explained_variance_ratio_




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
pca_score_df = pd.concat([pca_score_df,csv_input['class']],axis=1)
#2次元主成分空間上に口コミ（クラスタ）ベクトルをプロットする。また、上記で求めた主成分の値が大きい語彙特徴量も同時にプロットする
#ax = pca_score_df.plot(kind='scatter', x='PC2', y='PC1',style=['ro', 'bs'],s=5, alpha=0.2, figsize=(40,10))
current_palette = sns.color_palette(n_colors=4)
sns.scatterplot(x='PC2', y='PC1',  data=pca_score_df,hue='class', alpha = 0.7, s=100, palette=current_palette)
#上記と同じ平面上に、主成分値の大きい語彙をプロットする

c=0
for word, pca2,pca1 in zip(feature_pca_df['words'],feature_pca_df['PC2'],feature_pca_df['PC1']):
    #語彙のベクトルのプロットに語彙ラベルをアノテーションする
    plt.arrow(0,0,pca2*4.5,pca1*4.5,width=0.002,head_width=0.01,head_length=0.04,length_includes_head=True,color='blue')
    plt.annotate(word,xy=(pca2*5.5,pca1*5.),size=14,color='blue')
    
    
plt.show()

            
