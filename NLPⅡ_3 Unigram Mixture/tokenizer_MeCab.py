import MeCab
import pandas as pd
import unicodedata
import re

def tokenize(text):
    
    
    
    tagger = MeCab.Tagger("mecabrc")
    #tagger = MeCab.Tagger("mecabrc")
    node = tagger.parseToNode(text)

    tokens = []
    last_noun=0
    noun_phrase=''

    while node:            
        '''
        features = node.feature.split(',')
        if node.surface != '' :
            tokens.append(node.surface)
        '''
        features = node.feature.split(',')
        if node.surface != '':
            if features[0] =='名詞' or features[0] =='形容詞' :
                tokens.append(node.surface)
            #elif  features[0] =='動詞' :
            #    tokens.append(features[6])            
        
        node = node.next

    
    return tokens   
            
    
    
