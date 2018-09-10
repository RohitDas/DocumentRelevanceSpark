"""
    Class loads the big data file as a SQLContext object
"""

from pyspark import SparkConf,SparkContext, SQLContext

class DataLoader(object):
    def __init__(self, 
                 spark_context):
        self.spark_context = spark_context
        self.sql_context_created = False

    def visualize(self, sqlrdd):
        #print schema of the sql rdd
        sqlrdd.printSchema();

    def load_data_under_cur_context(self, 
                                    file_path,
                                    sample_factor=1):
        """
            The function returns an RDD of the form (DOCID, DOCSTRING)
            DOCID: reviewerId + asin
            DOCSTRING: reviewText|summary
        """
        if not self.sql_context_created:
            sql_context = SQLContext(self.spark_context)
            self.sql_context_created = True
            
        book_reviews = sql_context.read.json(file_path)

        self.visualize(book_reviews)
        book_reviews_mod = book_reviews.select("reviewText", 
                                               "summary",
                                               "reviewerId", 
                                               "asin").rdd.map(lambda x: ("".join(x[2:])," ".join(x[0:2])))

        if sample_factor < 1:
            return book_reviews_mod.sample(False, sample_factor)
        else:
            return book_reviews_mod
    
        
def get_new_spark_context(conf):
        return SparkContext(conf=conf)


if __name__ == "__main__":
    conf = SparkConf().setAppName("TF-IDF")
    context = get_new_spark_context(conf)
    data_loader = DataLoader(context)
    file_path = "/home/rohittulu/Downloads/bookreviews.json" #Placed in the same directory for now.
    bookreviews = data_loader.load_data_under_cur_context(file_path)
    print(bookreviews.take(10))
