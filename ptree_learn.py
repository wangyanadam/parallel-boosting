import sys

from pyspark import SparkContext
from pyspark import SparkConf
from operator import add

def __main__():
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: parallel boosting training <file>"
        exit(-1)

    #conf = SparkConf()
    #conf.set("spark.master","spark://hadoop33.rutgers.edu:7077")
    #conf.set("spark.app.name", "parallel-boosting")
    #conf.set("spark.executor.memory", "2g")
    #conf.set("spark.python.worker.memory", "4g")
    #conf.set("spark.local.dir", "/filer/tmp1/yw298/spark/tmp")
    #conf.set("spark.shuffle.memoryFraction", 0.4)
    #conf.set("spark.shuffle.consolidateFiles", "true")
    #conf.set("spark.io.compression.codec", "lzf")
    #conf.set("spark.shuffle.manager", "SORT")

    executor_memory = "512m"
    worker_memory = "512m"
    memory_fraction = "0.2"

    for arg in sys.argv:
        words = arg.split("=")
        if words[0] == "--executor_memory":
            executor_memory = words[1]
            print (">>> executor_memory = " + executor_memory)
        elif words[0] == "--worker_memory":
            worker_memory = words[1]
            print (">>> worker_memory = " + worker_memory)
        elif words[0] == "--memory_fraction":
            memory_fraction = words[1]
            print (">>> memory_fraction = " + memory_fraction)

    properties = [\
            ("spark.master", "spark://hadoop33.rutgers.edu:7077"),\
            ("spark.app.name", "parallel-boosting"),\
            ("spark.executor.memory", executor_memory),\
            ("spark.local.dir", "/filer/tmp1/yw298/spark/tmp"),\
            ("spark.python.worker.memory", worker_memory),\
            ("spark.shuffle.memoryFraction", memory_fraction),\
            ("spark.shuffle.consolidateFiles", "true"),\
            ("spark.io.compression.codec", "lzf"),\
            ("spark.shuffle.manager", "SORT"),\
            ]
    conf = SparkConf().setAll(properties)


    sc = SparkContext(conf=conf)

    lines = sc.textFile(sys.argv[1], 1)
    counts = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(add)
    
    output = counts.collect()
    for (word, count) in output:
        print "%s: %i" % (word, count)

    sc.stop()

if __name__ == "__main__":
    __main__()
