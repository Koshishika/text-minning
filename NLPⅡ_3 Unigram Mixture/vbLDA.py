
import pandas as pd
import gensim
from gensim import corpora, models
import codecs
import topic_model_vb2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
import seaborn as sns
matplotlib.pyplot.rcParams['figure.figsize'] = (16.0, 10.0)
 
#with open('./text/article.pickle', 'rb') as f:
#    review_text = pickle.load(f)
#csv_input = pd.read_csv('article_sampled2.csv', encoding='ms932', sep=',')
#dict = gensim.corpora.Dictionary.load_from_text('./dict_corpus/article.dict')


with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
	
with open('dict_inv.pickle', 'rb') as f2:
    dict_inv = pickle.load(f2)
dict={i:w for w,i in dict_inv.items()}

dict_sort=sorted(dict.items(), key=lambda x: x[0])
midashi =  [id_word[1] for id_word in dict_sort]
#print(dict_inv)
        
#print(dataset)
#input()
dataset = [ doc for doc in dataset if len(doc)!=0]
passes=40
no_of_topics =5
lda = topic_model_vb2.topic_model(no_of_topics,dataset,dict_inv)
theta,phi,p,p_byTopic = lda.lda_vb(passes)



topic_id = range(no_of_topics)
topic_vector = pd.DataFrame(phi,columns = midashi, index = topic_id)
with open('topic_vector.pickle', 'wb') as f20:
    pickle.dump(topic_vector,f20)




word_prob_dict={}
word_prob_matrix=[]
for id,word_dist_inTopic in enumerate(phi):
    #確率の小さいものは除いて辞書を参照する
    for i,word_prob in enumerate(word_dist_inTopic):
        #辞書を作成してvalueの降順に並べる
        word_prob_dict[dict[i]]=word_prob
    word_prob_list=sorted(word_prob_dict.items(), key=lambda x: -x[1])
    print('topic id:',id)
    print(word_prob_list[0:20])
    #word_prob_matrix.append(word_prob_list)
    word_prob_matrix.append(word_prob_list[0:30])

word_prob_df = pd.DataFrame(word_prob_matrix,columns = list(range(30)))
with codecs.open("topc_vectorsVB.csv", "w", "ms932", "ignore") as file2:    
    word_prob_df.to_csv(file2, index=False, encoding="ms932", mode='w', header=True)


#print('minimum_perplexity :' , p.min())
#print('minimum_preplexity by topic :',p_byTopic[passes-1])
x=np.arange(passes)
plt.figure(figsize = (21,12))
plt.plot(x,p)
plt.show()
#plt.savefig('./perplexity/article.png')

'''
thetaは2次元配列（行　文書数　列　トピック数）
これをもともとのarticle_sampled.csvにconcat
日付毎にgroupbyして、縦計を入れると、日付別、トピック別の頻度（確率値）が得られる
'''



