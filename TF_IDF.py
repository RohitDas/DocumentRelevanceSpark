"""
    Class calculates the TF_IDF of the document
"""
from data_loader import DataLoader, get_new_spark_context
from vocab_creator import Preprocessor, VocabularyCreator
import os, re
from pyspark import SparkContext, SparkConf

STOPWORDS = set()

def load_stopwords(stopwords_fp):
    if not os.path.exists(stopwords_fp):
        print("Error! Stopwords file doesn't exist. ")
    with open(stopwords_fp, 'r') as fp:
        stopwords = map(lambda x: x.strip(), fp.readlines())
    return set(stopwords)


def tokenize(token_string):
    tokens = filter(lambda x: x!='', re.split("\\W+", token_string.lower()))
    return list(set(tokens).difference(STOPWORDS))

class TFIDFCalcular(object):
    def __init__(self,
                 context,
                 data_path,
                 stopwords_path,
                 sample_factor):
        self.context = context
        self.data_path = data_path
        self.stopwords_path = stopwords_path
        self.sample_factor = sample_factor
        self.is_objs_initialized = False

    def initialize_objs(self):
        self.data_loader = DataLoader(self.context)
        self.preprocessor = Preprocessor(stopwords_path)
        self.vocab_creator = VocabularyCreator(self.preprocessor)
        self.is_objs_initialized = True

    def get_cartesian_rdd(self, 
                          docids,
                          vocab):
        assert docids != None, "Error caused due to null docids"
        assert vocab != None, "Error caused due to null vocab"
        return docids.cartesian(vocab)

    def calculate_tf(self,
                     doc_ids_to_doc_str):
        doc_to_tokens_rdd = doc_ids_to_doc_str.map(lambda x: (x[0], tokenize(x[1])))
        tfs = doc_to_tokens_rdd.flatMapValues(lambda x: x).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y)
        return tfs

    def calculate_df(self,
                      doc_ids_to_doc_str):
        df = doc_ids_to_doc_str.map(lambda x: (x[0],tokenize(x[1]))).flatMapValues(lambda x: x).distinct().map(lambda (title,word): (word,title)).countByKey()
        return df

    def calculate_tf_idf(self):
        if not self.is_objs_initialized:
            self.initialize_objs()
        doc_ids_to_doc_str = self.data_loader.load_data_under_cur_context(self.data_path,
                                                                          self.sample_factor)
        docids, vocab = self.vocab_creator.get_vocab(doc_ids_to_doc_str)

        cartesian_docs = self.get_cartesian_rdd(docids,
                                                vocab)
        tf = self.calculate_tf(doc_ids_to_doc_str)
        df = self.calculate_df(doc_ids_to_doc_str)


        return tf

        
if __name__ == "__main__":
    conf = SparkConf().setAppName("TFIDF")
    context = SparkContext(conf=conf)
    data_path = "/home/rohittulu/Downloads/bookreviews.json"
    stopwords_path = "/home/rohittulu/Downloads/stopwords.txt"
    sample_factor = 0.0001
    STOP_WORDS = load_stopwords(stopwords_path)
    tf_idf_calculator = TFIDFCalcular(context,
                                      data_path,
                                      stopwords_path,
                                      sample_factor)

    cartesian_docs = tf_idf_calculator.calculate_tf_idf()
    print cartesian_docs.take(200)
