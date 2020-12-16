#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:12:50 2020

@author: russell
"""
from SupervisedAlgorithm import Logistic_Regression_Classifier
from SupervisedAlgorithm import  Performance
from SupervisedAlgorithm  import TF_IDF
from pandas import read_excel
import numpy as np

def get_data_n_label_n_predcition(excel_file, sheet_name):

    sheet_name = sheet_name.strip()
    data = read_excel(excel_file, sheet_name = sheet_name)
  
    content = data.values

    X = content[:,0]
    Y = content[:,1]
    Z = content[:,2]

    
    Y = Y.astype('int')
    Z = Z.astype('int')
    
  
    return X, Y,Z


#------------SsentiA----used in the paper-------  
def apply_SSSentiA(): 
 
    dataset = "imdb" #"dvd" #"electronics"  #"electronics" #"book" #"dvd" #"book"
  
    # Very-high Confidence Group (Bin 1)
    X_1, Y_1,Z_1 = get_data_n_label_n_predcition("/Users/russell/Documents/NLP/Paper-2-SSentiA/Final/" + dataset + "_1.xlsx", dataset + '_1')
    
    # High Confidence Group (Bin 2)
    X_2, Y_2,Z_2 = get_data_n_label_n_predcition("/Users/russell/Documents/NLP/Paper-2-SSentiA/Final/" + dataset + "_2.xlsx", dataset + '_2')
    
    # Low Confidence Group (Bin 3)
    X_3, Y_3,Z_3 = get_data_n_label_n_predcition("/Users/russell/Documents/NLP/Paper-2-SSentiA/Final/" + dataset + "_3.xlsx", dataset + '_3')
    
    # very-Low Confidence Group (Bin 4)
    X_4, Y_4,Z_4 = get_data_n_label_n_predcition("/Users/russell/Documents/NLP/Paper-2-SSentiA/Final/" + dataset + "_4.xlsx", dataset + '_4')
   
    # Zero Confidence Group  (Bin 5)
    X_5, Y_5,Z_5 = get_data_n_label_n_predcition("/Users/russell/Documents/NLP/Paper-2-SSentiA/Final/" + dataset + "_5.xlsx", dataset + '_5')
    

    ml_classifier = Logistic_Regression_Classifier() 
    #lr_classifier = SVM_Classifier()
    #r_classifier = RandomForest_Classifier()
    #lr_classifier = RandomForest_Classifier()
    #lr_classifier = MultinomialNB_Classifier()
    #lr_classifier = ExtraTree_Classifier()
    
    
    bin_size_1_2 = len(X_1) + len(X_2) # + len(X_3) #+  len(X_01) + len(X_02) + len( X_11) + len(X_12)
    print("---",bin_size_1_2)
    
    
    data = np.concatenate((X_1,X_2,X_3), axis=None)
    label = np.concatenate((Z_1,Z_2,Y_3), axis=None)

  
    
    tf_idf = TF_IDF()
    data = tf_idf.get_tf_idf(data)
    
    X_train = data[:bin_size_1_2]
    Y_train = label[:bin_size_1_2]
    
    X_test = data[bin_size_1_2:]
    Y_test = label[bin_size_1_2:]
    

    
    prediction_bin_3 = ml_classifier.predict(X_train, Y_train, X_test)
    
    print("Bin-3 Results")
    performance = Performance()
    _,precision,  recall, f1_score, acc = performance.get_results(Y_test, prediction_bin_3)
    print("Total: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
    

    data = np.concatenate((X_1,X_2,X_3,X_4,X_5), axis=None)
    label = np.concatenate((Z_1,Z_2,prediction_bin_3,Y_4,Y_5), axis=None)
    
    
    tf_idf = TF_IDF()
    data = tf_idf.get_tf_idf(data)

    bin_1_2_3_training_data = len(X_1) + len(X_2) + len(X_3)  
    
    X_train = data[:bin_1_2_3_training_data]
    Y_train = label[:bin_1_2_3_training_data]
    
    X_test = data[bin_1_2_3_training_data:]
    Y_test = label[bin_1_2_3_training_data:]
    
 
    print("Bin-4results")
    prediction_bin_4_5 = ml_classifier.predict(X_train, Y_train, X_test)
    _,precision,  recall, f1_score, acc = performance.get_results(Y_test[:len(X_4)], prediction_bin_4_5[:len(X_4)])
    print("F1: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )
     

    print("Bin 5 results")
    _,precision,  recall, f1_score, acc = performance.get_results(Y_test[len(X_4):], prediction_bin_4_5[len(X_4):])
    print("F1: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )

    
    
    true_label = np.concatenate((Y_1,Y_2,Y_3,Y_4,Y_5), axis=None)
    all_prediction = np.concatenate((Z_1,Z_2,prediction_bin_3, prediction_bin_4_5), axis=None)
    
    print("\nOverall Predcition of SSSentiA")
    #precision,  recall, f1_score, acc = performance.get_results(true_label, all_prediction)
    _,precision,  recall, f1_score, acc = performance.get_results(true_label, all_prediction)
    print("Overall: ", round(precision,4),  round(recall,4), round(f1_score,4),round(acc,4) )

  
def main():
    apply_SSSentiA()
    
if __name__ == main():
    main()
