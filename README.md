# SSentiA
Code for SSentiA and LRSentiA




This repository contains code for SSentiA and LRSentiA (We are still adding/cleaning up code).

LRSentiA is a lexicon-based classifier used for sentiment analyis. It is capable of classifying in both binary and ternary level.  

LRSentiA utilizes Opinion Lexicon (Hu and Liu, KDD-2004). https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html. We have included the lexicon in this repository. 




The following libraries are required-

1. Scikit-learn https://scikit-learn.org/stable/ (For machine learning classifier) 

2. Spacy  https://spacy.io. (For parsingg and POS tag assignment for LRSentiA classifier)

3. Numpy

4. Panda



The steps are as follows-

1. Use LRSentiA (LRSentiA.py) to generate pseudo-label.
2. Use SSentiA (SSentiA.py) to train machine learning classifier utilizing the  pseudo-label training data. 


Please email me at ssazz001@odu.edu for any query.






