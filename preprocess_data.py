import sys
import numpy as np
import random as rnd
import pickle as pk

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel

def __main__():
    # Get program options
    input = ""
    nparts = 1
    test_ratio = 0.1
    test_output = ""
    train_output = ""
    ycol = 0
    batch_size = 10240
    save_rdd = 1 
    fs = "file:"
    fsep = ","
    normalize = 1

    for option in sys.argv:
        opt_val = option.split('=')
        if opt_val[0] == '--input':
            input = str(opt_val[1])
        elif opt_val[0] == '--normalize':
            normalize = int(opt_val[1])
        elif opt_val[0] == '--test_ratio':
            test_ratio = float(opt_val[1])
        elif opt_val[0] == '--test_output':
            test_output = str(opt_val[1])
        elif opt_val[0] == '--train_output':
            train_output = str(opt_val[1])
        elif opt_val[0] == '--nparts':
            nparts = int(opt_val[1])
        elif opt_val[0] == '--ycolumn':
            ycol = int(opt_val[1])
        elif opt_val[0] == '--batch_size':
            batch_size = int(opt_val[1])
        elif opt_val[0] == '--save_rdd':
            save_rdd = int(opt_val[1])
        elif opt_val[0] == '--fs':
            fs = str(opt_val[1])
        elif opt_val[0] == '--feature_sep':
            if opt_val[1] == 't':
                fsep = '\t'
            elif opt_val[1] == 'c':
                fsep = ','
            else:
                print >> sys.stderr, "Delimiter: wrong format, neither <tab> nor <,>"
                exit(-1)

    # Initialize Spark
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Mapping function for distinguishing between testing data (key = 0) and training data (key = 1)
    def func_pmap(p_iter):
        rnd.seed()
        
        for yx in p_iter:
            vals = np.fromstring(string = yx, sep = fsep)
            yval = vals[ycol]
            xvec = np.delete(vals, ycol)

            coin = rnd.random()
            if coin <= test_ratio:
                yield (0, (yval, xvec))
            else:
                yield (1, (yval, xvec))

    # Take test and train data
    data = sc.textFile(input, nparts) \
            .mapPartitions(func_pmap) \
            .persist(StorageLevel.MEMORY_AND_DISK)

    test_data = data.filter(lambda kv : kv[0] == 0).values() \
            .persist(StorageLevel.MEMORY_AND_DISK)
    train_data = data.filter(lambda kv : kv[0] == 1).values() \
            .persist(StorageLevel.MEMORY_AND_DISK)

    def func_max_train(yx1, yx2):
        z = []
        x1 = yx1[1]
        x2 = yx2[1]
        for e1, e2 in zip(x1, x2):
            z.append(max(e1, e2))

        return (0, np.array(z))

    if normalize:
        max_train_data = train_data.reduce(func_max_train)
        train_data = train_data.map(lambda (y, x) : (y, x / max_train_data[1]))
        test_data = test_data.map(lambda (y, x) : (y, x / max_train_data[1]))

        # Save the normalizers
        of = open(train_output + '.normalizers', 'w')
        pk.dump(max_train_data[1], of)
        of.close()

    # Save marked data
    if save_rdd:
        test_data.saveAsPickleFile(path = fs + test_output, batchSize = batch_size)
        train_data.saveAsPickleFile(path = fs + train_output, batchSize = batch_size)
    else:
        of = open(test_output, 'w')
        pk.dump(test_data.collect(), of)
        of.close()

        of = open(train_output, 'w')
        pk.dump(train_data.collect(), of)
        of.close()

if __name__ == '__main__':
    __main__()
