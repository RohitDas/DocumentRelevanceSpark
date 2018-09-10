"""
    Class calculates the TF_IDF of the document
"""
from data_loader import DataLoader, get_new_spark_context
from vocab_creator import Preprocessor, VocabularyCreator

from pyspark import SparkContext, SparkConf

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

    def calculate_tf_idf(self):
        if not self.is_objs_initialized:
            self.initialize_objs()
        doc_ids_to_doc_str = self.data_loader.load_data_under_cur_context(self.data_path,
                                                                          self.sample_factor)
        docids, vocab = self.vocab_creator.get_vocab(doc_ids_to_doc_str)

        cartesian_docs = self.get_cartesian_rdd(docids,
                                                vocab)
        return cartesian_docs

        
if __name__ == "__main__":
    conf = SparkConf().setAppName("TFIDF")
    context = SparkContext(conf=conf)
    data_path = "/home/rohittulu/Downloads/bookreviews.json"
    stopwords_path = "/home/rohittulu/Downloads/stopwords.txt"
    sample_factor = 0.001
    tf_idf_calculator = TFIDFCalcular(context,
                                      data_path,
                                      stopwords_path,
                                      sample_factor)

    cartesian_docs = tf_idf_calculator.calculate_tf_idf()
    print cartesian_docs.take(200)
