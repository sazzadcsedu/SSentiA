#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:48:29 2019

@author: russell
"""

import csv
import spacy
import nltk
import numpy as np

nlp_english = spacy.load('en')
from nltk.stem import PorterStemmer

dic = {}

class LexicalAnalyzer(object):
    
    def __init__(self):
        self.create_polarity_dictionary_opinion_lexicon()
        
        
    def read_data(self, excel_file, sheet_name):

        sheet_name = sheet_name.strip()
        data = read_excel(excel_file, sheet_name = sheet_name)
        #data = pd.read_csv("/Users/russell/Documents/NLPPaper/Comments.csv") #text in column 1, classifier in column 2.
        numpy_array = data.values
       # print(numpy_array)
        X = numpy_array[:,0]
        Y = numpy_array[:,1]
        Z = numpy_array[:,2]
       
       
        return X, Y,Z


    def preprocess_data(self,text):

        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.strip().replace(".", ".").replace(".", ".")
        text = text.lower()
        return text


    def stem_data(self, text):
        ps = PorterStemmer() 
        modified_text = " ".join(ps.stem(w)for w in nltk.wordpunct_tokenize(text))  

        return modified_text

    def split_review_text(self,review):
        import re
        split_sentences = review.split("�")
        split_sentences  = re.split('[.,]', review) # re.split('[^a-zA-Z][]', review)
        sentences = []
   
        for s in split_sentences:
            if len(s) > 1:
                sentences.append(s)
       
        return sentences

    def remove_pronoun(self,tokens):
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'who','me', 'him', 'her', 'it', 'us', 'you', 'them', 'whom','mine', 'yours', 'his', 'hers', 'ours', 'theirs','this', 'that', 'these', 'those']
    
        for token in tokens:
            for pronoun in pronouns:
                if token == pronoun:
                    #print("^^^^^^^^\:   ", token, pronoun)
                    tokens.remove(token)
                    break
                
                
        return tokens

    #----------- Check whether text contains adjective and adverb --------------                
    def does_contain_adjective(self, tokens):
        
        text = ' '.join(tokens)

        tokens = nlp_english(text)
        for token in tokens:
            if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
                return True
     
        return False

    #----------- Check polarity Shifter due to Negation  --------------  
    def get_negation_score(self,tokens):
      
        text = ' '.join(tokens)
 
        tokens = nlp_english(text)
        for i in range(len(tokens) - 2):
            if tokens[i].dep_ == 'neg' and  (tokens[i + 1].pos_ == 'ADJ' or tokens[i + 2].pos_ == 'ADJ' ):# or token.text == 'should' or token.text == 'could' or token.text == 'must' :
               
               if tokens[i + 1].pos_ == 'ADJ':
                   token = tokens[i + 1].text
               else:
                   token =  tokens[i + 2].text
               
               if token in  dic.keys(): 
                   return  - 2 * dic[token]
               
        #not good, do not like, not terrible    
        for i in range(len(tokens) - 1):
             if tokens[i].dep_ == 'neg' and  (tokens[i + 1].pos_ == 'ADJ'):# or token.text == 'should' or token.text == 'could' or token.text == 'must' :
               
                token = tokens[i + 1].text
              
                if token in  dic.keys(): 
                   #print("^^^^^ ^^^ ^ ^^  ^^ " , token, dic[token])
                   return  - 2 * dic[token]
        return 0
    
    
    #----------- Check presence of Comparison in sentence  --------------  
    def get_comparison_score(self,tokens):
        text = ' '.join(tokens)
        #print(text)
        tokens = nlp_english(text)
        for i in range(len(tokens) - 2 ):
            token = tokens[i]
            next_token = tokens[i + 2]
            #could/should/must be ?,  negate the ?
            if token.text == 'should' or token.text == 'could' or token.text == 'must' :
                if next_token in dic.keys(): 
                    return   dic[next_token] * -1
        return 0
    
    
    #----------- Bing Liu opinion Lexicon --------------  
    def create_polarity_dictionary_opinion_lexicon(self):
        file = open('/Users/russell/Documents/NLP/resource/positive.txt', 'r') 
        for line in file: 
            token = line.split()
            key = ''.join(token)
            dic[key] = 1
            
        file = open('/Users/russell/Documents/NLP/resource/negative.txt', 'r') 
        for line in file: 
            token = line.split()
            key = ''.join(token)
            dic[key] = -1
      
    
    def get_polarity_score(self, aspect_sentence):
        total_sum = 0
        positive_score = 0
        negative_score = 0
        ddd = 0
        for token in aspect_sentence:
            if token in dic.keys(): 
                total_sum += dic[token]
                if dic[token] == 1:
                    positive_score += dic[token]
                elif dic[token] == -1:
                    negative_score  -= dic[token]
                else:
                    ddd = 0
        
        return total_sum, positive_score, negative_score

    
    def remove_text_index(self,text):
        
        text = text.strip()
        index = text.find(":")
        text = text [index + 1:]
        
        return text
    

    def classify_binary_dataset(self, X_data, Y_label):
        
        num_of_detection = 0
        true_prediction = 0
        false_prediction = 0
        
        num_of_high_confidence = 0
        numOfZeroScore = 0;
        
        very_high_confidence = []
        zero_confidence = []
        high_confidence = []
        prediction_confidence_scores = []
        
        prediction = []
        i = 0
       
        
        
        for user_review in X_data:
            #print(i, user_review)
            if len(str(user_review)) < 5:
                i += 1
                prediction_confidence_scores. append(-1)
                print("\n\n\n\n: Less: ",user_review, "\n\n\n\n" )
                continue;
            
            sentiments = []
            
            user_review = self.preprocess_data(user_review)
            user_review = self.split_review_text(user_review)
            
            
            total_score = 0
            total_aspect_term = 0
            total_positive_score = 0
            total_negative_score = 0
            
            for sentence in user_review:  
                tokens = nlp_english(sentence)
                #print(">> ",tokens)
                aspect_sentence = []
                #text = word_tokenize(sentence)
                #print("## ##",nltk.pos_tag(text))
             
                for token in tokens:   
                    # if not token.is_stop:
                    #if  token.dep_ == 'nsubj' or  token.dep_ == 'amod' or token.pos_ == 'ADJ':
                    if token.dep_ == 'nsubj' or token.dep_ == 'neg'  or token.dep_ == 'advmod' or token.dep_ == 'ROOT' or token.dep_ == 'compound' or token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.text == 'could' or token.text == 'must' or token.text == 'should':
                       #print(token.text, token.dep_,  token.pos_,  [child for child in token.children])
                       aspect_sentence.append(token.text)
                
                #print("aspect_sentence",aspect_sentence)
                if len(aspect_sentence) >= 2:
                    num_of_detection += 1
                
                    sentiments.append(aspect_sentence)
                    
                    aspect_sentence = self.remove_pronoun(aspect_sentence)
                    
                    
                    if self.does_contain_adjective(aspect_sentence) == True:
                        #removed += 1
                       # print("*******" , aspect_sentence)
                        
                        score, positive,negative = self.get_polarity_score(aspect_sentence)
                        #print("-- ",i, score, positive,negative)
                        
                        total_positive_score += positive
                        total_negative_score += negative
                        #print ("Score:   ", score)
                       # score += get_polarity_score_sentic_net(aspect_sentence)
                        
                        negation_score = self.get_negation_score(aspect_sentence)
                        if abs(negation_score) > 0 :  
                            score  +=  negation_score
                            if negation_score < 0:
                                total_positive_score -= 1
                                total_negative_score += 1
                            else:
                                total_negative_score -= 1
                                total_positive_score += 1
                                
                            #print ("Neg: ",score,  aspect_sentence, negation_score)
                        total_score += score
                        total_score += get_comparison_score(aspect_sentence)
                        total_negative_score -=  get_comparison_score(aspect_sentence)
                        #if (total_score == 0):
                            #total_score = Vader_Lexicon(aspect_sentence)
                    
                #("##### Total Score:  ", total_score ) 
                  
                total_aspect_term += len(aspect_sentence)
            
            '''
            if total_score == 0:
                numOfZeroScore += 1
                zero_confidence.append(i)
            '''   
                
            
           
            #print("numOfZeroScore : ", numOfZeroScore)
            predicted_label = 0
            true_label = int(Y_label[i])
        
            if total_score >= 0:
                predicted_label = 1 
                
            
            
            
            total_positive_negative = total_positive_score + total_negative_score
            
            print(i, total_positive_score, total_negative_score, total_positive_negative, total_score)
            
            if total_score != 0:
                #if total_positive_score >= total_negative_score:
                prediction_confidence_score =  float(abs(total_positive_score - total_negative_score)/total_positive_negative)
            else:
                prediction_confidence_score = 0
            prediction_confidence_scores.append(prediction_confidence_score)
            
            
            #if total_score > (float(total_aspect_term) * 0.15) or total_score < -(float(total_aspect_term) * 0.15):
                
            #if (total_score != 0 and total_positive_negative != 0  and (float(total_positive_score/total_positive_negative) > 0.75 or float(total_negative_score/total_positive_negative) > 0.75 )):
               
            if prediction_confidence_score >= 0.5:
                very_high_confidence.append(i)
            elif total_score != 0:
            #total_score == 1 or total_score == -1:
                num_of_high_confidence += 1
                high_confidence.append(i)
            else: 
                #print("*************```",total_score)
                #print("---------- ", i, user_review , Y_label[i])
                numOfZeroScore += 1
                zero_confidence.append(i)
                
            #if total_score > 1 or total_score < -1:
            #if total_score > 2 or total_score < -2:
            #if total_score > len(user_review) or total_score < -len(user_review) :
           
            
            
            '''   
            if int(Y_label[n])> 2:
                true_label = 5
                
            '''
            if predicted_label == true_label:
                true_prediction += 1
            else:
                false_prediction += 1
                for aspect_sentence in sentiments:
                    score = self.get_polarity_score(aspect_sentence)
                   # print( aspect_sentence , score)
                    
               # print("\n\n\n  @@@ : ", true_label , predicted_label, total_score)
               
            prediction.append(predicted_label)
                
            
            i += 1
        #        if n == 3000:
        #            break
        
        print("\nLexicon: ")
        #avg_precision_recal(Y_test, prediction)
        results_f1_p_r = []
        
        print("\n!!!!!!",len(Y_label), len(prediction))
        
        f1score, precision,  recall = avg_precision_recal(Y_label, prediction)
        print("----> ",f1score)
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/results_process/" + current_dataset + "/" + str(process_id) + ".txt")
        
        #write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/dvd/" + str(process_id) + ".txt")
           
        write_label_prediction_to_file(X_data,  Y_label , prediction, prediction_confidence_scores, "/Users/russell/Downloads/electronics/" + str(process_id) + ".txt")
           
          

    
    def classify_ternary_dataset(self,data_english):
        
        prediction_confidence_scores = []
        
        polarity_scores = []
        positive_scores = []
        negative_scores = []
        
        polarity_orientation = []
        
        i = 0
        
        
        for user_review in data_english:
            #print(i, user_review)
            sentiments = []
            user_review = self.preprocess_data(user_review)
            #user_review = stem_data(user_review)
            user_review = self.split_review_text(user_review)
            #tokens = [token.text for token in s if not token.is_stop]
            
            total_score = 0
            if i % 50 == 0:
                print("-------------------------------------  ", i, len(user_review))
                
            #print("-------------------------------------  ", i, user_review , len(user_review))
            total_aspect_term = 0
            total_positive_score = 0
            total_negative_score = 0
            
            for sentence in user_review:  
                tokens = nlp_english(sentence)
                #print(">> ",tokens)
                aspect_sentence = []
                #text = word_tokenize(sentence)
                #print("## ##",nltk.pos_tag(text))
             
                for token in tokens:   
                    # if not token.is_stop:
                    #if  token.dep_ == 'nsubj' or  token.dep_ == 'amod' or token.pos_ == 'ADJ':
                    if token.dep_ == 'nsubj' or token.dep_ == 'neg'  or token.dep_ == 'advmod' or token.dep_ == 'ROOT' or token.dep_ == 'compound' or token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.text == 'could' or token.text == 'must' or token.text == 'should':
                       #print(token.text, token.dep_,  token.pos_,  [child for child in token.children])
                       aspect_sentence.append(token.text)
                
                #print("aspect_sentence",aspect_sentence)
                if len(aspect_sentence) >= 1:
                
                    sentiments.append(aspect_sentence)
                    
                    aspect_sentence = self.remove_pronoun(aspect_sentence)
                    
                    
                    if self.does_contain_adjective(aspect_sentence) == True:
                        #removed += 1
                       # print("*******" , aspect_sentence)
                        
                        score, positive,negative = self.get_polarity_score(aspect_sentence)
                        #print("-- ",i, score, "Pos: ",positive, "Neg: ",negative)
                        
                        total_positive_score += positive
                        total_negative_score += negative
                        #print ("Score:   ", score)
                       # score += get_polarity_score_sentic_net(aspect_sentence)
                        
                        negation_score = self.get_negation_score(aspect_sentence)
                        if abs(negation_score) > 0 :  
                            score  +=  negation_score
                            if negation_score < 0:
                                total_positive_score -= 1
                                total_negative_score += 1
                            else:
                                total_negative_score -= 1
                                total_positive_score += 1
                                
                            #print ("Neg: ",score,  aspect_sentence, negation_score)
                        total_score += score
                        total_score += self.get_comparison_score(aspect_sentence)
                        total_negative_score -=  self.get_comparison_score(aspect_sentence)
                        #if (total_score == 0):
                            #total_score = Vader_Lexicon(aspect_sentence)
                    
                #("##### Total Score:  ", total_score ) 
                  
                total_aspect_term += len(aspect_sentence)
            
            i += 1
                
           # print("$$$$ ----total score : ", total_score, predicted_label)
            
            polarity_scores.append(total_score)
            positive_scores.append(total_positive_score)
            negative_scores.append(total_negative_score)
            
            prediction_confidence = 0
            if total_score != 0:
                total_positive_negative = total_positive_score + total_negative_score
                prediction_confidence =  float(abs(total_positive_score - total_negative_score)/total_positive_negative)
           
            prediction_confidence_scores.append(prediction_confidence)
            
            
        return positive_scores,negative_scores, polarity_scores, prediction_confidence_scores
          
def main():
    
    lexicalAnalyzer = LexicalAnalyzer()
    
    data,label = lexicalAnalyzer.get_data_n_label()
    
    lexicalAnalyzer.classify_binary_dataset()
    
    
    print(" ")
         
if __name__ == main():
    main()