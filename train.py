#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code sample taken from the following gensim tutorial: Accessed 2016-04-01
# Parameters for w2v training were changed
# http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp,max_sentence_length=1000000), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count(),negative=15,sg=1)

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    model.save(outp)