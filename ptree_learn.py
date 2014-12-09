import sys

from pyspark import SparkContext
from pyspark import SparkConf
from operator import add

def func_map_partition(part_iter):
    pid = 0
    for x in part_iter:
        kv_pair = (pid, x)
        pid+=1
        yield kv_pair

def __main__():
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: parallel boosting training <file>"
        exit(-1)

    # Initialize Spark
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Create text-file RDDs
    train_data = sc.textFile(sys.argv[1], 3)
    mapped_data = train_data.mapPartitions(func_map_partition)
    reduced_data = mapped_data.groupByKey()

    result = reduced_data.collect()

    for (key, words) in result:
        pword = ""
        for word in words:
            pword += str(word)
        print(pword)
    
    #output = counts.collect()
    #for (word, count) in output:
        #print "%s: %i" % (word, count)

    sc.stop()

if __name__ == "__main__":
    __main__()
