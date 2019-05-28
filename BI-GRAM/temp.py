# -*- coding: utf-8 -*-
"""
Spyder Editor
@author:anuj
This is a temporary script file.
"""

import pandas as pd
import io
#t=open('en_US.txt', "r", encoding='utf-8')
t = io.open("disk1.txt", 'r', encoding='ISO-8859-1')
txt=t.read()
    
t.close()
from collections import OrderedDict
import re
review = re.sub('[^a-zA-Z]',' ',txt)
review= review.split()
review= [word.lower() for word in review]
#review= review.lower()
#review = [[words for words in review.lower()]
uni= OrderedDict()
uni= [review[i] for i in range(len(review)-1)]
uni=tuple(uni)
#import collections
#uni=collections.Counter([(x) for (x) in uni])
#uni = uni.most_common(len(uni))

word_pair = [(review[i],review[i+1]) for i in range(len(review)-1)]
l= len(word_pair)
gram2= set(word_pair)
coll_list = []
import collections
val_1=collections.Counter([(x,y) for (x,y) in word_pair])
val_1= val_1.most_common(l)
li=tuple(val_1)

li1= [item[1] for item in li]
el= [item[0] for item in li]
el=tuple(el)
li2= [item[0] for item in el]
li4= [item[1] for item in el]
conditionalprob = OrderedDict()
li1= [item[1] for item in li]

for i in range(len(li1)):
    li3=li1[i]
    firstword= li2[i]
    secondword= li4[i]
    #data= collections.Counter([(firstword) for (firstword) in uni])
    data= uni.count(firstword)
    cprob= li3/int(data)
    conditionalprob[firstword+" "+secondword]= cprob
   
    
check= input("Enter a word  to predict(exit to stop):")

    
matched= []
for i in range(len(li1)):
    if check==li2[i]:
        matched.append(li2[i]+" "+li4[i])
    
import heapq
topDict = {}
for singleBigram in matched:
    topDict[singleBigram] = conditionalprob[singleBigram]
    
topBigrams = heapq.nlargest(5, topDict, key=topDict.get)

for b in topBigrams:
    print (b+" : "+str(topDict[b])+"\n")
     
     
    
    
    
        
     
