"""
    This class takes an PipeLinedRDD of the (DOCID, DOCSTRING) and creates a vocab of words.
"""

class Preprocessor(object):
    pass

class VocabularyCreator(object):
    def __init__(self, 
                 preprocessor):
        self.preprocessor = preprocessor

    def get_vocab(self, docid_to_docstring_rdd):
        #Tokenize the documents
        tokenized_doc = docid_to_docstring_rdd.map(lambda x: self.preprocessor.tokenize(x[1]))
        #Undergo furhter preprocessing
        preprocessed_doc = tokenized_doc.map(lambda x: self.preprocessor.preprocess(x))
        #Take the distinct words
        vocab = preprocessor_doc.distinct()
        return vocab






