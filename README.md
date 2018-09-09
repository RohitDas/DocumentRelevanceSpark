# DocumentRelevanceSpark

Tasks:

Document representation creation:

Create the document pipelined RDD
Create the vocabulary(removing vocab)
Create a cartesian product of doc to word
Map into (key, 0) where key is made by hash(doc, word)
Read the doc
Calculate TF
Calculate DF, store it in a python collection
Calculate TF_IDF 
Merge 2 RDDs
Sort the result
Aggregrate by key to get the document vector representation of the docs


Querying:
Using Cosine similarity.
