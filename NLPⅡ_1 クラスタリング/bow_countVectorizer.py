import janome
from janome.tokenizer import Tokenizer
import pandas as pd
t = Tokenizer()




def make_corpus(documents):
  result_corpus=[]
  for adocument in documents:
    words=[token for token in t.tokenize(adocument, wakati=True)]
    text=" ".join(words)
    result_corpus.append(text)
  return result_corpus

document1='私は秋田犬が大好き。秋田犬は私が大好き。'
document2='私は犬が少し苦手。'
documents=[document1, document2]
corpus=make_corpus(documents)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
#一文字の単語を許容するように、token_patternを指定する
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out(corpus))
print(X.toarray())


vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(2, 2))
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out(corpus))
print(X.toarray())


vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 2)) 
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out(corpus))
print(X.toarray())