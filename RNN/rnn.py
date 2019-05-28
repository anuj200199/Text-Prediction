"""
Spyder Editor
@author: Anuj
This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train=open('x.txt' , encoding='ISO-8859-1')
training_set=dataset_train.read()
dataset_train.close()
import re
training_set=re.sub('[^a-zA-Z]',' ',training_set)
training_set=training_set.lower()
import nltk
import nltk.tokenize
nltk_tokens = nltk.word_tokenize(training_set)
#tokens=nltk_tokens
print (nltk_tokens)
tokens=list(set(nltk_tokens))
tokens=sorted(tokens)
"""for word in nltk_tokens:
    if word not in nltk_tokens:
        tokens.append(nltk_tokens)
"""
n_vocab=len(tokens)
words_to_int=dict((i, c) for c,i in enumerate(tokens))
int_to_words=dict((i,c) for i,c in enumerate(tokens))
#tf-idf(w, d) = bow(w, d) * N / (# documents in which word w appears)
tokens=np.array(tokens)
n=len(tokens)
dataX = []
dataY = []
seq_length=60
n_chars=len(nltk_tokens)
for i in range(60, n_chars, 1):
	seq_in = nltk_tokens[i-60:i]
	seq_out = nltk_tokens[i]
	dataX.append([words_to_int[char] for char in seq_in])
	dataY.append(words_to_int[seq_out])
""""for i in range(60,n):
    X_train.append(nltk_tokens[i-60:i])
    y_train.append(nltk_tokens[i])
X_train=np.array(X_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
"""
n_patterns=len(dataX)
X=np.array(dataX)
X = np.reshape(dataX, (n_patterns, seq_length,1))
#X=np.reshape(X, (X.shape[0],X.shape[1],1))
# normalize
X = X / float(n)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras import optimizers
#from keras.layers import Embeddings
y = to_categorical(dataY)
regressor=Sequential()
regressor.add(LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(256, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(256))
regressor.add(Dropout(0.2))
regressor.add(Dense(100, activation='relu'))
regressor.add(Dense(y.shape[1], activation='softmax'))
print(regressor.summary())
#optimizer =optimizers. RMSprop(lr=0.01)
#regressor.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#regressor.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#regressor.fit(X,y,epochs=10,batch_size=28)

#regressor.save('model.h5')
#regressor.fit(X, y, epochs=50, batch_size=28)
length=10
ix = [np.random.randint(n)]
y_char = [int_to_words[ix[-1]]]
X = np.zeros((length, n,1))
for i in range(length):
    X[0, i, :][ix[-1]] = 1
    print(int_to_words[ix[-1]], end="")
    ix = np.argmax(regressor.predict(X[:, :i+1, :])[0], 1)
    y_char.append(int_to_words[ix[-1]])
generate=('').join(y_char)



initial_text=input("Enter: ")
initial_text = [words_to_int[c] for c in initial_text]
GENERATED_LENGTH = 5
test_text = initial_text
generated_text = []

for i in range(5):
    X_X = np.reshape(test_text, (1, seq_length , 1))
    next_character = regressor.predict(X_X/float(n))
    index = np.argmax(next_character)
    generated_text.append(int_to_words[index])
    test_text.append(index)
    test_text = test_text[1:]

seed_text= input("Enter a word  to predict(Exit to stop):")
print(seed_text + '\n')
result=list()
in_text=seed_text

for i in range(5):
    encoded=words_to_int[in_text]
