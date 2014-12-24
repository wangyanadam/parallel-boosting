import sys
import random as rnd
import string as st

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD

import numpy as np
import pickle as pk
from sklearn import tree

def __main__():
    # Get program options
    input_path = ""
    num_learners = 1
    num_parts = 1
    output_path = '/filer/tmp1/yw298/spark/output/'
    fs = 'file:'
    save_data = 0

    # Parameters for base learner
    max_depth = None
    max_features = None
    min_samples_leaf = 1
    min_samples_split = 2

    # Parameters for coefficient fitting
    regularizer = None
    niters = 100
    reg_weight = 1.0
    step_size = 1.0
    batch_frac = 1
    
    for option in sys.argv:
        opt_val = option.split('=')
        if opt_val[0] == '--input':
            input_path = str(opt_val[1])
        elif opt_val[0] == '--fs':
            fs = str(opt_val[1])
        elif opt_val[0] == '--num_learners':
            num_learners = int(opt_val[1]) - 1
        elif opt_val[0] == '--num_parts':
            num_parts = int(opt_val[1])
        elif opt_val[0] == '--max_depth':
            max_depth = int(opt_val[1])
        elif opt_val[0] == '--max_features':
            max_features = int(opt_val[1])
        elif opt_val[0] == '--min_samples_leaf':
            min_samples_leaf = int(opt_val[1])
        elif opt_val[0] == '--min_samples_split':
            min_samples_split = int(opt_val[1])
        elif opt_val[0] == '--output':
            output_path = str(opt_val[1])
        elif opt_val[0] == '--regularizer':
            regularizer = str(opt_val[1])
        elif opt_val[0] == '--niters':
            niters = int(opt_val[1])
        elif opt_val[0] == '--reg_weight':
            reg_weight = float(opt_val[1])
        elif opt_val[0] == '--step_size':
            step_size = float(opt_val[1])
        elif opt_val[0] == '--batch_fraction':
            batch_frac = float(opt_val[1])
        elif opt_val[0] == '--save_data':
            save_data = int(opt_val[1])

    print '>>> input_path = %s' % str(input_path)
    print '>>> num_learners = %s' % str(num_learners)
    print '>>> num_parts = %s' % str(num_parts)
    print '>>> output_path = %s' % str(output_path)
    print '>>> max_depth = %s' % str(max_depth)
    print '>>> max_features = %s' % str(max_features)
    print '>>> min_samples_leaf = %s' % str(min_samples_leaf)
    print '>>> min_samples_split = %s' % str(min_samples_split)
    print '>>> regularizer = %s' % str(regularizer)
    print '>>> niters = %s' % str(niters)
    print '>>> reg_weight = %s' % str(reg_weight)
    print '>>> step_size = %s' % str(step_size)
    print '>>> file_system = %s' % str(fs)
    print '>>> batch_fraction = %s' % str(batch_frac)
    print '>>> save_data = %s' % str(save_data)

    if input_path == "":
        print >> sys.stderr, "Usage: parallel boosting training <file>"
        exit(-1)

    # Initialize Spark
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Map function of mapping training data to each learner (1):
    # Randomly partitioning the entire dataset.
    def func_pmap_rndpartition(p_iter):
        rnd.seed()

        for (k, v) in p_iter:
            yval = v[0]
            xvec = v[1]

            kv_pair = (rnd.randint(1, num_learners + 1), (yval, xvec))
            yield kv_pair

    # Map function of mapping training data to each learner (2):
    # Mapping each example to num_learners copies with their labels corrupted
    # with standard Gaussian multiplications.
    def func_pmap_rndlabeling(p_iter):
        rnd.seed()

        for v in p_iter:
            yval = v[0]
            xvec = v[1]

            # Emitting the training set with true labels
            yield (0, (yval, xvec))

            # Emitting the training sets with corrupted labels
            for tid in range(num_learners):
                coin = rnd.random()
                if coin < 0.5:
                    yval_rnd = -yval
                else:
                    yval_rnd = yval

                kv_pair = (tid + 1, (yval_rnd, xvec))
                yield kv_pair

    # Def of mapping function for training each learner
    def func_train_learner((key, data)):
        yvec = []
        xmat = []

        # Emitting the trained learners
        for (yval, xvec) in data:
            # Append label and feature values
            xmat.append(xvec)
            yvec.append(yval)

        # Train learner
        learner = tree.DecisionTreeRegressor( \
                max_depth = max_depth, \
                max_features = max_features, \
                min_samples_leaf = min_samples_leaf, \
                min_samples_split = min_samples_split \
                )
        learner.fit(xmat, yvec)

        return (key, learner)

    # Hypothesis sampling
    train_data_HS = sc.pickleFile(fs + input_path) \
            .repartition(num_parts) \
            .persist(StorageLevel.MEMORY_AND_DISK)

    train_data_map_HS = train_data_HS.mapPartitions(func_pmap_rndlabeling).\
            combineByKey(createCombiner = lambda v : [v], \
            mergeValue = lambda c, v : c + [v], \
            mergeCombiners = lambda c1, c2 : c1 + c2 \
            )

    learner_class = train_data_map_HS.map(func_train_learner).collect()
    learner_class_broadcast = sc.broadcast(learner_class)

    # Map function of generating training data for coefficient fitting
    def func_map_ypredmat(v):
            yval = v[0]
            xvec = v[1]

            values = np.zeros(num_learners + 1)
            for l in learner_class_broadcast.value:
                values[l[0]] = l[1].predict(xvec)

            return LabeledPoint(yval, values)

    # Coefficient fitting
    train_data_CF = train_data_HS \
            .map(func_map_ypredmat) \
            .persist(StorageLevel.MEMORY_AND_DISK)

    coeffs = LinearRegressionWithSGD.train(\
            data = train_data_CF, \
            iterations = niters, \
            step = step_size, \
            regType = regularizer, \
            regParam = reg_weight, \
            miniBatchFraction = batch_frac, \
            intercept = False
            )

    coeffs_list = sorted(list(enumerate(coeffs.weights, start = 1)), \
            key = lambda kv : kv[0])

    learner_class_list = sorted(learner_class, \
            key = lambda kv : kv[0])

    if save_data:
        # Save the raw training data
        train_data_HS.saveAsPickleFile(path = fs + output_path + '/train_data', batchSize = 10240)

        # Save the coeff-fit training data
        train_data_CF.saveAsPickleFile(path = fs + output_path + '/coeff_data',  batchSize = 10240)

    # Save the learner class and fitted coefficients
    file = open(output_path + '/learner_class', 'w')
    pk.dump(learner_class_list, file)
    file.close()

    file = open(output_path + '/fitted_coeffs', 'w')
    pk.dump(coeffs_list, file)
    file.close()

    sc.stop()

if __name__ == "__main__":
    __main__()
