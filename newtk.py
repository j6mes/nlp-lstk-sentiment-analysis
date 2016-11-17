#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from functools import partial
from dataset_reader import getData
from sklearn.metrics import classification_report
from sklearn import svm

import numpy as np
import random
import sys
import multiprocessing
import copy_reg
import types

from kernels import Kernels, model
from sentiment import class_functions

random.seed(5834682)

#Shuffle pairs of data (training examples and labels)
def shuffle(x,y):
    data = []
    for i in range(0,len(x)):
        data.append([x[i], y[i]])

    random.shuffle(data)

    x = []
    y = []

    for i in range(0,len(data)):
        x.append(data[i][0])
        y.append(data[i][1])

    return x,y


#This serializer is needed to send a class function as a message for worker threads
#http://stackoverflow.com/questions/25156768/cant-pickle-type-instancemethod-using-pythons-multiprocessing-pool-apply-a
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)






#Set the classifier function based on the program arguments
getClassOf = class_functions[int(sys.argv[1])]


#Load the dataset with the embeddings and the label type
X_train, y_train, X_test, y_test, X_dev, y_dev = getData(model,getClassOf)
sys.stdout.flush()


#Shuffle the data
X_train,y_train = shuffle(X_train,y_train)
X_dev,y_dev = shuffle(X_dev,y_dev)
X_test,y_test = shuffle(X_test,y_test)

#If we're training on a subset of the data. Bin the rest
if "subset" in sys.argv:
    X_train = X_train[:int(sys.argv[4])]
    y_train = y_train[:int(sys.argv[4])]


#This is a bit of a hack because the sklearn SVM kernel doesn't support strings as features.
# have to use a lookup table for each sentence id
X_train_ids = []
X_dev_ids = []
X_test_ids = []

for i in range(0,len(X_train)):
    X_train_ids.append([i])

for i in range(len(X_train),len(X_train)+len(X_dev)):
    X_dev_ids.append([i])

for i in range(len(X_train)+len(X_dev),len(X_train)+len(X_dev)+len(X_test)):
    X_test_ids.append([i])


#And get the sample from the lookup table
def getSample(id):
    if id<len(X_train):
        return X_train[id]
    elif id<len(X_train)+len(X_dev):
        return X_dev[id-len(X_train)]
    elif id<len(X_train)+len(X_dev)+len(X_test):
        return X_test[id-len(X_dev)-len(X_train)]



#Choose the kernel based on input arguments
k = Kernels(getSample)
kernels = [k.maxminavg,k.similarityKernel,k.similarityTreeKernel,k.optimisedSimilarityKernel,k.optimisedSimilarityTreeKernel]
kernel = kernels[int(sys.argv[2])]



#We need to pre-compute the gram matrix because of limitations with sklearn
#based off http://stackoverflow.com/questions/26962159/how-to-use-a-custom-svm-kernel
def proxy_kernel(X,Y,K):
    print "Pre-computing gram matrix:"

    nitr = len(X)*len(Y)
    done = 0
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if done % 100 == 0:
                print '{0:.2f}'.format(100*done/nitr)+"% done"
                sys.stdout.flush()
            gram_matrix[i, j] = K(x, y)
            done+=1
    return gram_matrix

#Doing it in parallel using a worker pool is a lot faster
def proxy_kernel_par(X,Y,K,pool):
    print "Pre-computing gram matrix:"
    nitr = len(X)
    done = 0
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        pk = partial(K,y=x)
        gram_matrix[i] = pool.map(pk,Y)
        done+=1
        print '{0:.2f}'.format(100*done/nitr)+"% done"
        sys.stdout.flush()
    return gram_matrix


#Get number of CPUs and set up a worker pool
cpus = multiprocessing.cpu_count()
p = multiprocessing.Pool(processes=cpus)


kernelProxy = partial(proxy_kernel_par, K=kernel,pool=p)





print "Training"
sys.stdout.flush()

cls = svm.SVC(kernel=kernelProxy,verbose=True)
cls.fit(X_train_ids,y_train)

print "Predicting"
predicted = cls.predict(X_test_ids)
print classification_report(y_test,predicted,digits=4)

p.close()
p.join()