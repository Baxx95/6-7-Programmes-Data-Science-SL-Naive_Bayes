# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:16:52 2021

@author: Zakaria
"""
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

smsdata = open('SMSSpamCollection.txt', 'r')
csv_reader = csv.reader(smsdata, delimiter='\t')

smsdata_data = []
smsdata_labels = []

for line in csv_reader:
    smsdata_labels.append(line[0])
    smsdata_data.append(line[1])
    
smsdata.close()

from collections import Counter

c = Counter(smsdata_labels)
print(c)

#=======================================================#
#================ PREPROCESSING NLP =====================
#=======================================================#

#from nltk import download, sent_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from nltk.stem  import PorterStemmer


#download('punkt')
#download('stopwords')
#download('averaged_perceptron_tagger')
#download('wordnet')

 
def Preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.sent_tokenize(sent)]
    # On met tout en minuscule
    tokens = [word.lower() for word in tokens]
    # On charge les stopword (les mots dénués de sens)
    stopwd = stopwords.words('english')
    # On charge seulement les mots qui ne sont les stopword dans la tokens
    tokens = [token for token in tokens if token not in stopwd]
    # On charge (enlève plutôt) les mot ayant moins de 4 caractères, car d'habitude pas de sens
    tokens = [word for word in tokens if len(word)>=3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)
    
    """
    NN,
    VB
    """
    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    
    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(tokens, tag):
        if tag in Noun_tags:
            # lemmatizer ne marche plus la nouvelle méthode qu'on utilise à ce jour est le WordNetLemmatizer()
            return lemmatizer.lemmatize(tokens, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(tokens, 'v')
        else:
            return lemmatizer.lemmatize(tokens, 'n')
    
    preprocess_text = " ".join([prat_lemmatize(token, tag) for token, tag in tagged_corpus])
    return preprocess_text
# FIN FONCTION PREPROCESSING

smsdata_data_2 = []

for i in smsdata_data:
    smsdata_data_2.append(Preprocessing(i))

#=======================================================#
#========== EVALUATION DU MODEL ET PREDICTION ===========
#=======================================================#

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(smsdata_data_2, smsdata_labels, test_size=.25, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2,
                             ngram_range=(1,2),
                             stop_words='english',
                             max_features=4000,
                             strip_accents='unicode',
                             norm='l2')
    
# LA FONCTION todense() permet de visualiser le resultat qui est un tableau de float
x_train_2 = vectorizer.fit_transform(x_train).todense()
x_test_2 = vectorizer.transform(x_test).todense() 

from sklearn.naive_bayes import MultinomialNB
y_test_NB_pred = MultinomialNB().fit(x_train_2, y_train).predict(x_test_2)

score_model_NB = MultinomialNB().fit(x_train_2, y_train).score(x_test_2, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_NB_pred, y_test)
    



