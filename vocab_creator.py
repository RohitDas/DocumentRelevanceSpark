"""
    This class takes an PipeLinedRDD of the (DOCID, DOCSTRING) and creates a vocab of words.
"""
import os
import re

class Preprocessor(object):
    def __init__(self, stopwords_fp):
        self.stopwords_fp = stopwords_fp
        self.is_stopwords_loaded = False

    def load_stopwords(self):
        if not os.path.exists(self.stopwords_fp):
            print("Error! Stopwords file doesn't exist. ")
        with open(self.stopwords_fp, 'r') as fp:
            stopwords = map(lambda x: x.strip(), fp.readlines())
        self.stopwords = set(stopwords)
    
    def tokenize(self, docstring):
        if not self.is_stopwords_loaded:
            self.load_stopwords()
            self.is_stopwords_loaded = True
        
        tokens = filter(lambda x: x!='', re.split("\\W+", docstring.lower()))
        return list(set(tokens).difference(self.stopwords))

    def preprocess(self, token):
        """
            The Preprocess logic will be added later
        """
        return token


class VocabularyCreator(object):
    def __init__(self, 
                 preprocessor):
        self.preprocessor = preprocessor

    def get_vocab(self, docid_to_docstring_rdd):
        #Tokenize the documents
        docids = docid_to_docstring_rdd.map(lambda x: x[0])
        tokenized_doc = docid_to_docstring_rdd.flatMap(lambda x: self.preprocessor.tokenize(x[1]))
        #Undergo furhter preprocessing
        preprocessed_doc = tokenized_doc.map(lambda x: self.preprocessor.preprocess(x))
        #Take the distinct words
        vocab = preprocessed_doc.distinct()
        return docids, vocab

if __name__ == "__main__":
    #Checking Preprocessor
    stopwords_fp = "/home/rohittulu/Downloads/stopwords.txt"
    preprocessor = Preprocessor(stopwords_fp)
    preprocessor.load_stopwords()
    print preprocessor.tokenize("Virat kohli played well but the other players sucked big time.")

    #Checking Vocabulary creator
    vocab_creator = VocabularyCreator(preprocessor)

    from data_loader import DataLoader, get_new_spark_context
    from pyspark import SparkConf
    conf = SparkConf().setAppName("TF-IDF")
    context = get_new_spark_context(conf)
    data_loader = DataLoader(context)
    file_path = "/home/rohittulu/Downloads/bookreviews.json" #Placed in the same directory for now.
    bookreviews = data_loader.load_data_under_cur_context(file_path, 0.001)
    docids, vocab = vocab_creator.get_vocab(bookreviews)
    print docids.take(20)
    print vocab.take(20)
