Implementation
==============

All code is tested and ran on the `teaching0` server using python/2.7.6. Dependencies: `numpy scipy sklearn gensim`

Data sets and pre-training
--------------------------

The embeddings were trained on the latest english wikipedia dump taken from <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2> taken on 1st April 2016. The total size of this corpus was 12Gb. The corpus was tokenized and had punctuation removed using gensim. Word stemming and lemmatization was not performed because these transformations are not present in the Stanford Sentiment Treebank dataset.

Run `corpus.py inputfile.bz2 outputfile`.

Word2vec word embeddings were generated using the gensim library with parameters: skipgram, size=300, context window = 5, negative sampling = 15, subsampling = 1e-3, min\_count=5.

Run `train.py corpus.txt wiki.model`

Sentiment data was taken from the Stanford Sentiment Treebank . This model was trained only on entire sentences and not on phrase fragments. For fine-grained sentiment analysis, we applied the same feature labels as . For polar sentiment analysis, we merge + with ++ labels and - with `–` and dropped neutral sentiment phrases.

SVM Implementation
------------------

The evaluation file can be run using the following command:

`newtk.py type kernel [subset N]`

Command Definition:

| Type |              | Kernel |                      |
|:-----|:-------------|:-------|:---------------------|
| 0    | Polar        | 0      | Baseline Min/Max/Avg |
| 1    | Fine-Grained | 1      | LSK with Embeddings  |
|      |              | 2      | LSTK with Embeddings |

A random subset of size N of the training data can be sampled with the optional parameter (subset N).

The code has the pre-requisites of the standard libraries: numpy, scipy, sklearn and gensim all of which are available on pip and have fast fortran or C subroutines.

Dataset Reader
--------------

The dataset reader (dataset\_reader.py) stores all training sentences and sentiment scores in memory. The TrainingSample class stores the sentence and a tree structure (tree.py) which is used to compute the depth feature weight.

The samples were randomised before training. The same seed was used in all experiments to ensure consistency.

SVM Solver
----------

The libsvm 1-vs-rest multi-class svm implementation was used. This library is included in the sklearn python package. There are two limitations with this software: 1) it does not natively support strings as features and 2) it requires the gram-matrix to be pre-computed.

To overcome the limitation with strings as features, two lookup tables were created (X\_train\_ids and X\_test\_ids), the id of the sentence was passed into the kernel function rather than the string literal. The kernel function has a pre-processing step that retrieves the string from the lookup table.

Pre-computing the gram matrix is an expensive process as pairwise similarity scores must be computed between every point in the data set. While this does add an up-front cost, this cannot be avoided given the current libraries available in python.To reduce the time taken to compute it, the gram matrix is computed in parallel using a number of worker threads.

The SVM decision boundary is typically defined using a sparse subset of support-vectors which means that there should be no need to compute the entire gram matrix. However, there is no guarantee over how many times distances between data points and the support vectors are computed. Pre-computing the gram matrix adds an up-front cost, while at the same reduces the cost of classification at runtime. The libsvm library is written in C and is highly optimised when provided with the gram matrix, whereas performing computations directly in python is considerably slower. While there only 10,000 samples, pre-computation of the gram matrix is achievable, but is bordering on infeasible. Although this technique will not scale to larger data sets, it is appropriate for this number of samples given the time constraints of the project and the resources available at the university.

To reduce the number of hits made to the word2vec embedding matrix for each convolution, we experimented with caching the embeddings. This made little noticeable difference.
