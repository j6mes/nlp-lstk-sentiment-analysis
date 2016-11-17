import numpy as np

from tree import tree

class TrainingSample:
    def __init__(self,model,sentence,ids):
        self.sentence = sentence
        self.tree = tree(model,ids,sentence)

    def words(self):
        return self.sentence.split(" ")

    def leaves(self):
        return self.tree.leaves()

def getData(model,getClassOf):
    splitsFile = open("datasetSplit.txt")
    phraseFile = open("datasetSentences.txt")
    dictionaryFile = open("dictionary.txt")
    sentimentFile = open("sentiment_labels.txt")
    treesFile = open("STree.txt")

    dictionary = dict()
    sentiments = dict()

    trees = dict()
    trainIds = []
    devIds = []
    testIds = []

    id = 1
    for line in treesFile.readlines():
        trees[id]=line.strip()
        id +=1

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
        dictionary[w.decode('utf-8').encode('ascii','ignore').strip()]=int(id)


    for line in sentimentFile.readlines():
        id,sentiment = line.strip().split("|")
        sentiments[int(id)] = float(sentiment)

    phrase = phraseFile.readline()


    train_sents = []
    train_classes = []

    test_sents = []
    test_classes = []

    dev_sents = []
    dev_classes = []

    nok = 0
    for line in phraseFile.readlines():
        pid,phrase = line.strip().split("\t")
        pid = int(pid)
        print line
        try:
            sentiment = sentiments[dictionary[phrase.decode('utf-8').encode('ascii','ignore').strip()]]
            ok = False
            for word in phrase.split(" "):
                word = word.lower()
                if word in model:
                    ok = True

            if ok and getClassOf(sentiment) is not None:
                nok += 1
                if pid in trainIds:
                    train_sents.append(TrainingSample(model,phrase,trees[pid]))
                    train_classes.append(getClassOf(sentiment))
                elif pid in testIds:
                    test_sents.append(TrainingSample(model,phrase,trees[pid]))
                    test_classes.append(getClassOf(sentiment))
                elif pid in devIds:
                    dev_sents.append(TrainingSample(model,phrase,trees[pid]))
                    dev_classes.append(getClassOf(sentiment))
        except Exception as e:
            print "Exception "+ str(e)


    print "TOTAL number of samples added"
    print nok

    print "Number of training sentences"
    print len(train_sents)

    print "Number of dev sentences"
    print len(dev_sents)

    print "Number of test sentences"
    print len(test_sents)

    return np.array(train_sents),np.array(train_classes),np.array(test_sents),np.array(test_classes),np.array(dev_sents),np.array(dev_classes)
