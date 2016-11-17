#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from gensim.models import word2vec
from sklearn.metrics import classification_report
import numpy as np
import sys

def getClassFineGrained(x):
    if x <= 0.2:
        return "NegNeg"
    elif x>0.2 and x<=0.4:
        return "Neg"
    elif x>0.4 and x<=0.6:
        return "Neu"
    elif x>0.6 and x<=0.8:
        return "Pos"
    elif x>0.8:
        return "PosPos"

def getClassPolar(x):
    if x <= 0.5:
        return "Neg"
    else:
        return "Pos"





kernels = ['poly','rbf','linear']
funcs = [getClassFineGrained, getClassPolar]
numclasses = [5,2]

kernel = kernels[int(sys.argv[1])]
fun = funcs[int(sys.argv[2])]
numclass = numclasses[int(sys.argv[2])]


print "Loading Model"

model = word2vec.Word2Vec.load('wiki.model')

print "Loading datasets"

splitsFile = open("datasetSplit.txt")
phraseFile = open("datasetSentences.txt")
dictionaryFile = open("dictionary.txt")
sentimentFile = open("sentiment_labels.txt")

dictionary = dict()
sentiments = dict()

trainIds = []
devIds = []
testIds = []

for line in splitsFile.readlines():
    id,split = line.strip().split(",")
    id = int(id)
    split = int(split)
    if split is 1:
        trainIds.append(id)
    elif split is 2:
        testIds.append(id)
    elif split is 3:
        devIds.append(id)



for line in dictionaryFile.readlines():
    w,id = line.strip().split("|")
    dictionary[w.decode('utf-8')]=int(id)


for line in sentimentFile.readlines():
    id,sentiment = line.strip().split("|")
    sentiments[int(id)] = float(sentiment)

phrase = phraseFile.readline()

train_embeddings = []
train_classes = []

test_embeddings = []
test_classes = []

dev_embeddings = []
dev_classes = []


for line in phraseFile.readlines():
    pid,phrase = line.strip().split("\t")
    pid = int(pid)
    linevecMax = None
    linevecMin = None
    linevecAvg = None
    done = 0
    try:
        words = phrase.decode('utf-8').split()
        sentiment = sentiments[dictionary[phrase.decode('utf-8')]]
        for word in words:
            print word
            word = word.lower()
            if word in model:
                if linevecMax is not None:
                    linevecMax = np.maximum(model[word],linevecMax)
                    linevecMin = np.minimum(model[word],linevecMin)
                    linevecAvg = np.add(model[word],linevecAvg)
                    done+=1
                else:
                    linevecMax = model[word]
                    linevecMin = model[word]
                    linevecAvg = model[word]
                    done+=1
        if linevecMax is not None:
            linevecAvg = np.divide(linevecAvg,done)
            linevec = np.concatenate((linevecMax,linevecMin,linevecAvg),axis=0)#,np.divide(linevecAvg,float(done))))
            if pid in trainIds:
                train_embeddings.append(linevec)
                train_classes.append(fun(sentiment))
            elif pid in testIds:
                test_embeddings.append(linevec)
                test_classes.append(fun(sentiment))
            elif pid in devIds:
                dev_embeddings.append(linevec)
                dev_classes.append(fun(sentiment))
    except Exception as e:
        print e


from sklearn import svm
cls = svm.SVC(C=numclass,kernel=kernel,degree=2)


x,y = train_embeddings, train_classes
cls.fit(x,y)


predicted = cls.predict(dev_embeddings)

print dev_classes
print predicted

print classification_report(dev_classes,predicted,digits=4)
