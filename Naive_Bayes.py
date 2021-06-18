# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:22:07 2019

@author: Formateur IT
"""
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import csv 

smsdata = open('SMSSpamCollection.txt','r') 
csv_reader = csv.reader(smsdata,delimiter='\t') 


smsdata_data = [] 
smsdata_labels = [] 

for line in csv_reader: 
    smsdata_labels.append(line[0]) 
    smsdata_data.append(line[1]) 
    
smsdata.close()

from collections import Counter 

c = Counter( smsdata_labels ) 
print(c)




import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import string 
import pandas as pd 
from nltk import pos_tag 
from nltk.stem import PorterStemmer

"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
"""

#Fonction a été écrit (pré-traitement) comprend toutes les étapes pour plus de commodité.

def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in 
            nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words('english') 
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word)>=3]
    stemmer = PorterStemmer() 
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)
    
    Noun_tags = ['NN','NNP','NNPS','NNS'] 
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ'] 
    lemmatizer = WordNetLemmatizer()
    def prat_lemmatize(token,tag): 
        if tag in Noun_tags: 
            return lemmatizer.lemmatize(token,'n') 
        elif tag in Verb_tags: 
            return lemmatizer.lemmatize(token,'v') 
        else: 
            return lemmatizer.lemmatize(token,'n')
        
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])              
    return pre_proc_text

smsdata_data_2 = [] 

for i in smsdata_data: 
    smsdata_data_2.append(preprocessing(i)) 
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(smsdata_data_2, smsdata_labels, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer 

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english',  
    max_features= 4000,strip_accents='unicode',  norm='l2')


x_train_2 = vectorizer.fit_transform(X_train).todense() 
x_test_2 = vectorizer.transform(X_test).todense()



from sklearn.naive_bayes import MultinomialNB 
clf = MultinomialNB().fit(x_train_2, y_train) 


ytest_nb_predicted = clf.predict(x_test_2) 
ytrain_nb_predicted = clf.predict(x_train_2) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, ytrain_nb_predicted)



    
    
    
    
    


























