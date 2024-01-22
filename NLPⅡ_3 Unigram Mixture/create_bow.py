from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tokenizer_MeCab import tokenize  # <1>
import codecs
import unicodedata
import re

fortravel = pd.read_csv('fortravel.csv', encoding='ms932', sep=',',skiprows=0)
texts = fortravel['body'].values.tolist()
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


#vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',min_df=0.05, max_df=0.3 )
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',min_df=0.005, max_df=0.5 )  ####
vec=vectorizer.fit(texts_list)  
bow = vectorizer.transform(texts_list)  # <4>
print(vec.vocabulary_)
print(bow)
print(bow.toarray())


fortravel_bow = pd.DataFrame(bow.toarray(), columns=vectorizer. get_feature_names_out())
assert len(fortravel_bow)==len(fortravel), 'len unmatch'
fortravel = pd.concat([fortravel, fortravel_bow], axis=1)
with codecs.open("fortravel_bow.csv", "w", "ms932", "ignore") as f: 
    #header=Trueで、見出しを書き出す
    fortravel.to_csv(f, index=False, encoding="ms932", mode='w', header=True)