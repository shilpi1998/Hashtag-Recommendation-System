# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:14:55 2018

@author: Rishika
"""

import keras
import json
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SimpleRNN, TimeDistributed 
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import string
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json


max_words = 1000 

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via', 'b', "b'RT"]
 
def tokenize(s):
    return tokens_re.findall(s)
    
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def stoping(s):
    x=[]
    for term in s:
        if term not in stop:
            x.append(term)
            count_all.update(x)
    return x

def convert_text_to_index_array(text):
    return [dictionary[word] for word in text]



#data extraction from an already saved csv file of twitter dataset
tweets=pd.read_csv('Two Thousand     .csv',sep=',')
data = tweets[['Tweet','Hashtag']]
data['Tweet'] = data['Tweet'].apply(lambda x: x.lower())
count_all= Counter()
data['Tweet']= [preprocess(tweet) for tweet in data['Tweet']]
data['Tweet']= [stoping(tweet) for tweet in data['Tweet']]
total=len(data['Tweet'])



#creating suitable input and output matrices for training the model
train_x = [tweet for tweet in data['Tweet']]
train_y1 = np.asarray([x for x in data['Hashtag'] ])
# create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)
# feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index
with open('dictionary2.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

allWordIndices = []
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
encoder = LabelEncoder()
encoder.fit(train_y1)
Y= encoder.transform(train_y1)
num_classes = np.max(Y) + 1
train_y = keras.utils.to_categorical(Y, num_classes)

Ydecode= encoder.inverse_transform(Y)
values_of_Y=dict(zip(Ydecode,Y))



a=list(train_x.shape) 
print(*a)
b=list(train_y.shape) 
print(*b)

def convert_text_to_index_array1(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices



def trainRNN():
    model = Sequential() 
    model.add(Embedding(max_words, 200, input_length=a[1] ))
    model.add(Dropout(0.2)) 
    model.add(SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2)) 
    model.add(Dense(250, activation='relu')) 
    model.add(Dropout(0.2)) 
    model.add(Dense(b[1] , activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit(train_x, train_y, batch_size=64, epochs=5)

    print('\nAccuracy: {}'. format(model.evaluate(train_x, train_y)[1]))


#saving the model for future use
    model_json = model.to_json()
    with open('RNNmodel.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('RNNmodel.h5')

    print('saved model!')

def predictOut():
    json_file = open('RNNmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights('RNNmodel.h5')

# okay here's the interactive part
    while 1:
        evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')

        if len(evalSentence) == 0:
            break

    # format your input for the neural net
        testArr = convert_text_to_index_array1(evalSentence)
        input1 = tokenizer.sequences_to_matrix([testArr], mode='binary')
        pred = model.predict(input1)
        indi= np.argmax(pred, axis=1)
        for i in Y:
            if (Y[i] == indi):
                print(i)
                print(Ydecode[i])
                break


while 1:
    process=int(input("Enter 1 to show output or 2 to to train:"))
    if(process==2):
        trainRNN()
    elif(process==1):
        predictOut()
    else:
        break
    


