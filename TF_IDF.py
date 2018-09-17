"""
    Class calculates the TF_IDF of the document
"""
from data_loader import DataLoader, get_new_spark_context
from vocab_creator import Preprocessor, VocabularyCreator
import os, re, math
from pyspark import SparkContext, SparkConf
from collections import OrderedDict

STOPWORDS = set()
VOCAB = None


def dot_prod(vector_a, vector_b):
    """
        Function calculates the dot product of 2 vectors.
    """
    vec_len = len(vector_a)
    dot_prod = 0
    for i in range(vec_len):
        dot_prod += vector_a[i]*vector_b[i]
    return dot_prod

def cosine_similarity_a(vector_a, vector_b):
    """
        Function calculates the cosine similarity.
    """
    return dot_prod(vector_a, vector_b)/(math.sqrt(dot_prod(vector_b, vector_b))*math.sqrt(dot_prod(vector_b, vector_b)))

def load_stopwords(stopwords_fp):
    if not os.path.exists(stopwords_fp):
        print("Error! Stopwords file doesn't exist. ")
    with open(stopwords_fp, 'r') as fp:
        stopwords = map(lambda x: x.strip(), fp.readlines())
    return set(stopwords)

def tokenize(token_string):
    tokens = filter(lambda x: x!='', re.split("\\W+", token_string.lower()))
    return list(set(tokens).difference(STOPWORDS))

def normalize(word_tf_idf_list):
    #Sum of squares
    S = math.sqrt(reduce(lambda x,y: x+y, map(lambda a: math.pow(a[1],2), word_tf_idf_list)))
    return map(lambda x: (x[0], x[1]/S), word_tf_idf_list)

def cosine_similarity_b(word_tf_idf_value, tokens):
    ordered_dict = OrderedDict()
    ordered_dict.update(word_tf_idf_value)
    vector_a, vector_b = [], []
    for token in tokens:
        if token in ordered_dict:
            vector_a.append(ordered_dict[token])
            vector_b.append(1)
    print vector_a, vector_b

    numerator = dot_prod(vector_a, vector_b)
    denominator = math.sqrt(dot_prod(ordered_dict.values(), ordered_dict.values()))
    denominator *= math.sqrt(len(tokens))
    return numerator/denominator

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
        self.doc_representation = None
        self.doc_to_str = None

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
        VOCAB = vocab.collect()
        cartesian_docs = self.get_cartesian_rdd(docids,
                                                vocab)
        tf = self.calculate_tf(doc_ids_to_doc_str)
        df = self.calculate_df(doc_ids_to_doc_str)
        
        total_docs = doc_ids_to_doc_str.count()
        #Calculating tf_idf
        tf_idf = tf.map(lambda x: (x[0], (1+math.log(df.get(x[0][1]))*math.log(total_docs/x[1]))))  
        #Sort according to the keys
        sorted_merged_tf_idf = tf_idf.sortByKey()
        #Group by key to form a vector
        representation = sorted_merged_tf_idf.map(lambda x: (x[0][0], (x[0][1], x[1]))).combineByKey(lambda x: [x], lambda u,v: u+[v], lambda u1, u2: u1+u2)
        #grouped_tf_idf = sorted_merged_tf_idf.groupByKey(lambda x: [x], lambda u,v: u+[v], lambda u1, u2: u1+u2)
        normalize_representation = representation.map(lambda x: (x[0], normalize(x[1])))
        self.doc_representation =  normalize_representation
        self.doc_to_str = doc_ids_to_doc_str

    def query(self, sentence):
        """
            Get the top k similar documents
        """
        tokens = tokenize(sentence)
        cosine_similarities_doc = self.doc_representation.map(lambda x: (cosine_similarity_b(x[1], tokens),x[0]))
        sorted_cosine_similarities_doc = cosine_similarities_doc.sortByKey(False)
        top_twenty = sorted_cosine_similarities_doc.take(20)
        return top_twenty

    def queries(self, sentences):
        """
        """
        for sentence in sentences:
            print self.query(sentence)
        
if __name__ == "__main__":
    conf = SparkConf().setAppName("TFIDF")
    context = SparkContext(conf=conf)
    #data_path = "/home/rohittulu/Downloads/bookreviews.json"
    data_path = "test_data_1.json"
    stopwords_path = "/home/rohittulu/Downloads/stopwords.txt"
    sample_factor = 1
    STOP_WORDS = load_stopwords(stopwords_path)
    tf_idf_calculator = TFIDFCalcular(context,
                                      data_path,
                                      stopwords_path,
                                      sample_factor)

    cartesian_docs = tf_idf_calculator.calculate_tf_idf()
    query_strings = ["touching fast moving plot"]
    top_twenty = tf_idf_calculator.queries(query_strings)
