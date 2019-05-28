# Text-Prediction
Predict the next word(s) considering the previous sequence of words.


In this project, I have developed 2 methods to predict the next word(s)

The first method is the Bigram model in which we have paired 2 words together and then calculated the occurence of the pair in the text.
The second model involves use of Deep Learning technique, that is building an Recurrent Neural Network.

Both of the methods are discuused below:

Method 1: Bi gram Model
a.) Input the name of file from which you want to train the model. In my project, i have used the file name disk1.txt which is uploaded.
b.) In order to improve accuracy, we have to perform certain operations on the text such as remove all the colons, commas, semicolons and other insignificant special characters which are of no use while predicting the next word, converting the text to lowercase, etc.
c.) As the name suggests, the next step is to form pairs of two consecutive words and count the times that these pair is occuring in the whole next.
d.) The idea is that higher the frequency of a particular pair in the document, higher are the chances of the second word in the pair to be output if the first word is taken as input. Therefore, we calculate probability of each pair in the document.
e.) Take user input and print the next common n words after the input. (In my py file, i have taken n=5).


Method 2: Recurrent Nueral Network: rnn.py file

a.) Input the name of file from which you want to train your model.(disk1.txt)
b.) Perform preprocessing on words such as lowercase conversion, removal of unwanted characters(,:;), tokenise the words.
c.) Find number of unique words and create 2 dictionaries
                         dict 1: word_to_int: convert the unique words(key) to integer values(value).
                         dict 2: int_to_word: each integer is mapped to a corresponding words(inverse of dict 1).
d.) Since RNN learns from sequence of previous words, we consider a sequence of 60 words, out of which the first 59 words in a tuple are sequence of words and the last word is the corresponding prediction. 
e.) Build an RNN model of youw own and input the training set into the model.
f.) Take user input as sequence of sentences and predict next words.


