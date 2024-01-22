from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import codecs
from tokenizer_MeCab import tokenize
import re
import unicodedata
import pickle
csv_input = pd.read_csv('fortravel.csv', encoding='ms932', sep=',',skiprows=0)
#texts = csv_input['text'].values.tolist()
texts = csv_input['body'].values.tolist()

stopwords_df = pd.read_csv('stopwords.csv', encoding='ms932', sep=',',skiprows=0)
stopwords=stopwords_df['stopwords'].tolist()
stopwords = list(set(stopwords))
# Bag of Words計算

texts_list=[]
for text in texts:
    text=unicodedata.normalize('NFKC',text)
    text=re.findall('[一-龥ぁ-んァ-ンー々]+',text )
    
    text= ''.join(text)
    
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]
    body = ' '.join(tokens)
    

    texts_list.append(body)


vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=0.005, max_df=0.5)  
#vectorizer = CountVectorizer(tokenizer=tokenize)
vec=vectorizer.fit(texts_list)  # <3>

dict_inv ={w:i for i,w in enumerate(vectorizer.get_feature_names()) if w !='' or type(w)!=float}
bow = vectorizer.transform(texts_list)  # <4>


#kana_filter = [k for k in bow.columns.tolist() if len(k)!=1 and len(re.findall('[ぁ-んァ-ン々]',k ))==0  ]
'''
kana_filter=[]
for k in bow.columns().tolist():
    if len(k)==1 and len(re.findall('[ぁ-んァ-ン々]',k )):
        kana_filter.append(k)
'''
#bow=bow.loc[:kana_filter]






#bowを単語列の行列に戻す
words_inDoc=vectorizer.inverse_transform(bow)
print(words_inDoc)
#w_vecs=[]
bow_filter =[]
word_vecs=[]
data_filter =[]

for w_vec,bow_vec,(i,data) in zip(words_inDoc, bow.toarray(), csv_input.iterrows()):
            
    #ひらがな1文字を削除する正規表現
    if len(w_vec)==0:
        continue
    #print(bow_vec)
    word_vec=[dict_inv[w] for w in w_vec if w in dict_inv]     
    #words= ','.join(word_vec)
    #w_vecs.append(word_vec)
   
    word_vecs.append(word_vec)
    bow_filter.append(bow_vec)
    data_filter.append(data.values)

assert len(word_vecs)==len(bow_filter) and len(word_vecs)==len(data_filter), 'unmach filter dataset'
with open('dataset.pickle', 'wb') as f:
    pickle.dump(word_vecs,f)    
with open('dict_inv.pickle', 'wb') as f:
    pickle.dump(dict_inv,f)

data_df = pd.DataFrame(data_filter,columns=csv_input.columns.tolist())
#with open('origin_data_filter.pickle', 'wb') as f:
#    pickle.dump(data_df,f)  
with codecs.open("origin_data_filter.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    data_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)


bow_filter_df = pd.DataFrame(bow_filter, columns=vectorizer.get_feature_names_out())
with open('bow.pickle', 'wb') as f:
    pickle.dump(bow_filter_df,f)
with codecs.open("fortravel_tokens.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    bow_filter_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)

'''
words_df = pd.DataFrame(w_vecs,columns=['words'])
con_df = pd.concat([csv_input,words_df],axis=1)
with codecs.open("fortravel_datasetLDA.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    con_df.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
'''

'''
print(vec.vocabulary_)
print(bow)
print(bow.toarray())

print('bow dim',len(vectorizer.get_feature_names()))
bow = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
tabelog_tokens = pd.concat([csv_input,bow],axis=1)
with codecs.open("fortravel_tokens.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    tabelog_tokens.to_csv(f, index=False, encoding="ms932", mode='w', header=True)
'''