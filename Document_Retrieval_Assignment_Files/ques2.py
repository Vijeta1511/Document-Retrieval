#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
------------------------------------------------------------
USE: python <PROGNAME> (options)
OPTIONS:
    -h : print this help message
    -s : use "with stoplist" configuration (default: without)
    -p : use "with stemming" configuration (default: without)
    -w LABEL : use weighting scheme "LABEL" (LABEL in {binary, tf, tfidf}, default: binary)
    -o FILE : output results to file FILE
------------------------------------------------------------\
"""
#==============================================================================
# Importing
import os, glob, re, sys, getopt
from collections import Counter
import numpy, time
import random, operator
import matplotlib.pyplot as plt

#==============================================================================
# Command line processing

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'ubt')
        opts = dict(opts)
        self.exit = True
        
        if '-u' in opts:
            self.runconfig(unigram)
            return
        
        if '-b' in opts:
            self.runconfig(bigram)
            return
        
        if '-t' in opts:
            self.runconfig(trigram)
            return
        else:
            self.runconfig(unigram)
            

start = time.time()
random.seed(876)


listofcounters = []
listofdictionaries = []
dictionaryofweights = {}
new_weights = []
accuracy = []
unigram = 1
bigram = 2
trigram = 3


pos_repository_path = 'review_polarity/txt_sentoken/pos'
neg_repository_path = 'review_polarity/txt_sentoken/neg'
        

def create_bag_of_words(file_path, label, ngram):
    repository = []
    file_lists = glob.glob(os.path.join(file_path , '*.txt'))
    for file_list in file_lists:
        dictionary = {}
        with open(file_list,'r') as f_input:
            text = f_input.read()
            counter = Counter(generate_ngrams(text, ngram))
            dictionary['words'] = counter
            dictionary['label'] = label
            
            listofdictionaries.append(dictionary)
            listofcounters.append(counter)
            repository.append(counter)
            
    for i in range(200):
        listofcounters.pop()
        
    return repository


def generate_ngrams(Text, ngram):
    # Convert to lowercases
    text = Text.lower()   
    # Replace all none alphanumeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
    # Break sentence in the token, remove empty tokens
    words = [token for token in text.split(" ") if token != ""]  
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    n_grams = zip(*[words[i:] for i in range(ngram)])
    listofngrams = [" ".join(n_gram) for n_gram in n_grams]
    return listofngrams
    

def create_weights_dictionary(repository):
    weights = {}
    for i in repository:    
        for keys, values in i.items():
            w = {keys : 0.0}
            weights.update(w)
    return weights


def perceptron(train_set, dictionaryofweights):
    for doc_dict in train_set:
        doc_score = []
        counts = doc_dict['words']
        label = doc_dict['label']
        for word, freq in counts.items():  
            word_score = freq * dictionaryofweights[word]
            doc_score.append(word_score)
        y_hat = numpy.sign(sum(doc_score))
        if y_hat == 0:
            y_hat = 1
            
        if y_hat != label:
            for word, freq in counts.items():
                if label == 1:
                    dictionaryofweights[word] += freq
                else:
                    dictionaryofweights[word] -= freq
                
    return dictionaryofweights 


def perceptron_train(train_set, test_set, dictionaryofweights):
    for i in range(16):
        random.shuffle(train_set)
        dictionaryofweights = perceptron(train_set, dictionaryofweights)
        accuracy_it = test_perceptron(test_set, dictionaryofweights)
        accuracy.append(accuracy_it)
        print("The accuracy for iteration ",i," is: ",accuracy_it)
        new_weights.append(dictionaryofweights.copy())
 
    return dictionaryofweights 


def average_weight(new_weights):
    sums = Counter()
    counters = Counter()
    for itemset in new_weights:
        sums.update(itemset)
        counters.update(itemset.keys())
        average_weight_dict = {x: float(sums[x])/counters[x] for x in sums.keys()}
    return average_weight_dict


def test_perceptron(test_set, dictionaryofweights):
    c = 0.0
    for doc_dict in test_set:
        doc_score = 0.0
        counts = doc_dict['words']
        label = doc_dict['label']
        for word, freq in counts.items():
            if word not in dictionaryofweights:
                continue
            doc_score += freq * dictionaryofweights[word]
        if doc_score >=0:
            if label == 1:
                c+=1
        else:
            if label == -1:
                c+=1   
    return(c/len(test_set))
    
    
def plot(accuracy):
    plt.plot(accuracy, marker='o', label='accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Prediction Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    
#==============================================================================
# MAIN

if __name__ == '__main__':

    config = CommandLine()
    if config.exit:
        sys.exit(0) 
           
    def runcongif(ngram):
        
        start = time.time()
        random.seed(876)
        
        
        listofcounters = []
        listofdictionaries = []
        dictionaryofweights = {}
        new_weights = []
        accuracy = []
        
        pos_repository = create_bag_of_words(pos_repository_path, 1, unigram)
        #print(pos_repository)
        
        neg_repository=create_bag_of_words(neg_repository_path, -1, unigram)
        #print(neg_repository[2])
        
        dictionaryofweights = create_weights_dictionary(listofcounters)
        ##print(dictionaryofweights)
        
        zero_weights_accuracy = test_perceptron(listofdictionaries, dictionaryofweights)
        print("The Accuracy with zero weights: ", zero_weights_accuracy,"\n")
        
        #Separating training and test sets
        pos_train = listofdictionaries[0:800]
        pos_test = listofdictionaries[800:1000]
        neg_train = listofdictionaries[1000:1800]
        neg_test = listofdictionaries[1800:2000]
        
        train_set = pos_train + neg_train
        test_set = pos_test + neg_test
        
        #calling perceptron algorithm and checking accuracy with single pass
        updated_weights = perceptron(train_set, dictionaryofweights)
        accuracy_before_training = test_perceptron(test_set, updated_weights)
        print("The Accuracy with before training model: ", accuracy_before_training,"\n")
        
        #calling perceptron algorithm and checking accuracy with multiple pass and random traning set
        updated_weights_train =perceptron_train(train_set,test_set, dictionaryofweights)
        accuracy_after_training = test_perceptron(test_set, updated_weights_train)
        print("\nThe Accuracy with after training model: ", accuracy_after_training,"\n")
        
        plot(accuracy)
        
        average_weight_dict = average_weight(new_weights)
        accuracy_on_average_weights = test_perceptron(test_set, average_weight_dict)
        print("\nThe Accuracy with average weights: ", accuracy_on_average_weights,"\n")
        
        #printing top 10 positive weighted words
        positive_sort = sorted(average_weight_dict.items(), key=operator.itemgetter(1))[::-1]
        print("Top 10 positive weighted words: ", positive_sort[:10],"\n")
        
        #printing top 10 positive weighted words
        negative_sort = sorted(average_weight_dict.items(),key=operator.itemgetter(1))
        print("\nTop 10 negative weighted words: ", negative_sort[:10],"\n")
        
        end = time.time()
        
        print(end-start)
    
    
