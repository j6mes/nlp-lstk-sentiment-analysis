#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code sample adapted from the following gensim tutorial: Accessed 2016-04-01
# http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

import sys
from gensim.corpora import WikiCorpus

corpus, out = sys.argv[1], sys.argv[2]
f = open(out, 'w')

wiki = WikiCorpus(corpus, lemmatize=False, dictionary={})

i = 0
for text in wiki.get_texts():
    f.write(" ".join(text) + "\n")
    i +=1
    if i % 1000 == 0:
        print "Done " + str(i)
f.close()
