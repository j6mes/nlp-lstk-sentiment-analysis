import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import word2vec
import sys


#First load the word2vec embeddings into memory
print "Loading Model"
model = word2vec.Word2Vec.load('wiki.model')
print "Done"
sys.stdout.flush()


class Kernels:
    def __init__(self,lookup_table):
        self.lookup_table = lookup_table


    #Hotswap our similarity measure
    def sim(self,x,y):
        #Cosine similarity
        if x in model and y in model:
            #Does scipy work faster for distance measures?
            #return np.dot(model[x],model[y])/np.sqrt(np.dot(model[x],model[x])*np.dot(model[y],model[y]))
            return 1 - cosine(model[x],model[y])
        else:
            #If neither object is in the grammar return 0
            return 0


    #For use in the baseline Kernel
    def embeddingForSentence(self,sentence):
        linevecMax = None
        linevecMin = None
        linevecAvg = None
        done = 0

        words = sentence.words()
        for word in words:
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

        linevecAvg = np.divide(linevecAvg,done)
        return np.concatenate((linevecMax,linevecMin,linevecAvg),axis=0)

    #The baseline kernel
    def maxminavg(self,x,y):
        w1 = x
        w2 = y

        t1 = self.lookup_table(int(w1[0]))
        t2 = self.lookup_table(int(w2[0]))
        w1vec = self.embeddingForSentence(t1)
        w2vec = self.embeddingForSentence(t2)

        return np.dot(w1vec,w2vec)/(np.sqrt(np.dot(w1vec,w1vec) * np.dot(w2vec,w2vec)))


    def lsk(self,x,y):
        t1 = self.lookup_table(int(x[0]))
        t2 = self.lookup_table(int(y[0]))

        sum = 0
        for a in t1.words():
            for b in t2.words():
               sum += self.sim(a.lower(),b.lower())
        return sum


    def ltsk(self,x,y):
        t1 = self.lookup_table(int(x[0]))
        t2 = self.lookup_table(int(y[0]))

        sum = 0
        for x_depth,x_word in t1.leaves():
            for y_depth,y_word in t2.leaves():
               sum += self.sim(x_word.getWord().lower(),y_word.getWord().lower()) * 1/x_depth * 1/y_depth
        return sum


    def similarityKernel(self,x,y):
        return self.lsk(x,y)/(np.sqrt(self.lsk(x,x) * self.lsk(y,y)))

    def similarityTreeKernel(self,x,y):
        return self.ltsk(x,y)/(np.sqrt(self.ltsk(x,x) * self.ltsk(y,y)))


    #We try a minor optimisation by flattening the normalisation in the kernel function. it makes negligable difference
    def optimisedSimilarityKernel(self,x,y):
        w1 = x
        w2 = y


        t1 = self.lookup_table(int(w1[0]))
        t2 = self.lookup_table(int(w2[0]))

        embeddings = dict()

        for depth,leaf in t1.leaves():
            if leaf.getWord().lower() not in embeddings and leaf.getWord().lower() in model:
                embeddings[leaf.getWord().lower()] = model[leaf.getWord().lower()]

        for depth,leaf in t2.leaves():
            if leaf.getWord().lower() not in embeddings and leaf.getWord().lower() in model:
                embeddings[leaf.getWord().lower()] = model[leaf.getWord().lower()]

        norm1 = 0
        for depth1,leaf1 in t1.leaves():
            for depth2,leaf2 in t1.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    norm1 += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()]))

        norm2 = 0
        for depth1,leaf1 in t2.leaves():
            for depth2,leaf2 in t2.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    norm2 += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()]))

        sum = 0
        for depth1,leaf1 in t1.leaves():
            for depth2,leaf2 in t2.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    sum += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()]))

        return sum/np.sqrt(norm1*norm2)


    #We try a minor optimisation by flattening the normalisation in the kernel function. it makes negligable difference
    def optimisedSimilarityTreeKernel(self,x,y):
        w1 = x
        w2 = y

        t1 = self.lookup_table(int(w1[0]))
        t2 = self.lookup_table(int(w2[0]))

        embeddings = dict()

        for depth,leaf in t1.leaves():
            if leaf.getWord().lower() not in embeddings and leaf.getWord().lower() in model:
                embeddings[leaf.getWord().lower()] = model[leaf.getWord().lower()]

        for depth,leaf in t2.leaves():
            if leaf.getWord().lower() not in embeddings and leaf.getWord().lower() in model:
                embeddings[leaf.getWord().lower()] = model[leaf.getWord().lower()]

        norm1 = 0
        for depth1,leaf1 in t1.leaves():
            for depth2,leaf2 in t1.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    norm1 += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()])) * 1/depth1 * 1/depth2

        norm2 = 0
        for depth1,leaf1 in t2.leaves():
            for depth2,leaf2 in t2.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    norm2 += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()])) * 1/depth1 * 1/depth2

        sum = 0
        for depth1,leaf1 in t1.leaves():
            for depth2,leaf2 in t2.leaves():
                if leaf1.getWord().lower() in embeddings and leaf2.getWord().lower() in embeddings:
                    sum += (1 - cosine(embeddings[leaf1.getWord().lower()],embeddings[leaf2.getWord().lower()])) * 1/depth1 * 1/depth2

        return sum/np.sqrt(norm1*norm2)

