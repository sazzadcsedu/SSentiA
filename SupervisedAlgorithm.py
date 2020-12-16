#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 18:47:48 2019

@author: russell
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from pycm import ConfusionMatrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier

import gzip
import gensim 
import logging


from nltk.sentiment.vader import SentimentIntensityAnalyzer 


import nltk
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
    

class TF_IDF(object):
    def __init__(self):
        print("..")
        
    def get_tf_idf(self, X):   
        vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=lambda x: x.split())
        X = vectorizer.fit_transform(X)
        #print(X)
        return X


class Performance(object):

    
    def get_results(self, labels, predictions):
        from sklearn.metrics import confusion_matrix
       
        
        
        conf_matrix = confusion_matrix(labels, predictions)
        
        #print(conf_matrix)
    
        precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 0)
        recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis = 1)
   
        accuracy =  np.sum(np.diag(conf_matrix) / np.sum(conf_matrix))
        
        #print("Total: " , np.sum(conf_matrix))
        
        #print("Correct/Incorrect : ", np.sum(np.diag(conf_matrix) ),  np.sum(conf_matrix) -  np.sum(np.diag(conf_matrix) ))
        #print("Denominator:", np.sum(conf_matrix, axis = 0))
        #print(np.mean(precision))
        #print(np.mean(recall))
    
        precision = np.mean(precision)
        recall = np.mean(recall)
 
        #print("Precision: ", precision)
        #print("Recall: " , recall)

    
        f1_score = (2 * precision * recall) / (precision + recall)
        #print("F-1 Score:  -----  ", f1_score)  
        return conf_matrix, precision,  recall, f1_score,  accuracy
    

    def calculateMCC(self, labels, predictions):

        cm = ConfusionMatrix(labels, predictions, digit=5)
        
        #print("\n Kappa-AC1: ", cm.Kappa, cm.AC1)
        #print("\nMCC:")
        #print(cm.MCC)
        return cm.MCC
    
    
    def calculate_roc_auc_score(self, labels, predictions):  
        score = roc_auc_score(labels, predictions)
        return score
        #print("ROC:" , score)


class KNN_Classifier(object):
    def predict(self, X_train, Y_train,  X_test):
        clf =   KNeighborsClassifier(n_neighbors=3) #svm.SVC( class_weight=  'balanced' , kernel = 'linear' ,max_iter=20000) #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        return prediction

class SGD_Classifier(object):
        
    def predict(self, X_train, Y_train,  X_test):
        clf =  SGDClassifier(loss="hinge", penalty="l2", max_iter=1500) #svm.SVC( class_weight=  'balanced' , kernel = 'linear' ,max_iter=20000) #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        return prediction


class SVM_Classifier(object):
        
    def predict(self, X_train, Y_train,  X_test):
        clf = svm.SVC( class_weight=  'balanced' , kernel = 'linear' ,max_iter=20000) #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        #clf = svm.SVC(C= 1.0, kernel = 'rbf' ,max_iter=20000, random_state = None)
        #clf = svm.SVC(kernel = 'rbf' ,max_iter=20000)
        #clf = svm.SVC(kernel = 'linear' ,max_iter=200000)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        return prediction
    
class LDA_Classifier(object): 
    def predict(self, X_train, Y_train,  X_test):
        clf = LinearDiscriminantAnalysis() #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        #clf = LogisticRegression()
        clf.fit(X_train, Y_train) 
        prediction = clf.predict(X_test)
        return prediction

    
class Logistic_Regression_Classifier(object):
    def predict(self, X_train, Y_train,  X_test):
        clf = LogisticRegression(class_weight= 'balanced') #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        #clf = LogisticRegression()
        clf.fit(X_train, Y_train) 
        prediction = clf.predict(X_test)
        return prediction
    
class Ridge_Regression_Classifier(object):
    def predict(self, X_train, Y_train,  X_test):
        clf = LogisticRegression(class_weight= 'balanced') #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", svm.SVC(C=1.0, kernel = 'linear',class_weight= {-1: 1.0, 0: 1.0, 1: 1.0} ,max_iter=10000))])
        #clf = LogisticRegression()
        clf.fit(X_train, Y_train) 
        prediction = clf.predict(X_test)
        return prediction
    
    
class Gradient_boosting_classifier(object):
    
    def predict(self,X_train, Y_train , X_test):
        model = XGBClassifier(class_weight= 'balanced')
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        return predictions
       
    

class MultinomialNB_Classifier:
    def predict(self, X_train, Y_train,  X_test):
        classifier =  MultinomialNB()  #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", MultinomialNB())])
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        return prediction

    
class RandomForest_Classifier(object):
    def predict(self,  X_train, Y_train, X_test):
   
        classifier = RandomForestClassifier(class_weight= 'balanced')  #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", )])
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        return prediction

class ExtraTree_Classifier(object):
    def predict(self, X_train, Y_train, X_test):
        classifier = ExtraTreesClassifier(random_state=100, class_weight= 'balanced') #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", ExtraTreesClassifier(random_state=0))])
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        return prediction
    
class AdaBoost_Classifier(object):
    def predict(self, X_train, Y_train, X_test):
        classifier = AdaBoostClassifier(n_estimators=100) #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", ExtraTreesClassifier(random_state=0))])
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        return prediction
   
    
class AdaBoost_Classifier(object):
    def predict(self, X_train, Y_train, X_test):
        classifier = AdaBoostClassifier(n_estimators=100) #Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", ExtraTreesClassifier(random_state=0))])
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        return prediction

class Multilayer_Perceptron_classifier:
    def __init__(X_train, Y_train , X_test):
        classifier = Pipeline([("vect", CountVectorizer(stop_words="english")),("tfidf", TfidfTransformer()),("clf", MLPClassifier(max_iter=10000))])
        classifier.fit(X_train, Y_train)
        print("MLPClassifier")
        prediction = classifier.predict(X_test)
        return prediction

