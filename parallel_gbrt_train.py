import sys
import random as rnd
import string as st

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LinearRegressionWithSGD

import numpy as np
import pickle as pk
from sklearn import tree

def __main__():
    # Get program options
    input_path = ""
    num_learners = 1
    num_parts = 1
    field_sep = '\t'
    output_path = '/filer/tmp1/yw298/spark/output/model.out'

    # Parameters for base learner
    max_depth = None
    max_features = None
    min_samples_leaf = 1
    min_samples_split = 2

    # Parameters for coefficient fitting
    regularizer = 'None'
    niters = 100
    reg_term = 1.0
    step_size = 1.0
    
    for option in sys.argv:
        opt_val = option.split('=')
        if opt_val[0] == '--input':
            input_path = str(opt_val[1])
        elif opt_val[0] == '--num_learners':
            num_learners = int(opt_val[1])
        elif opt_val[0] == '--num_parts':
            num_parts = int(opt_val[1])
        elif opt_val[0] == '--field_sep':
            if opt_val[1] == 't':
                field_sep = '\t'
            elif opt_val[1] == 's':
                field_sep = ','
            else:
                print >> sys.stderr, "Delimiter: wrong format, neither <tab> nor <,>"
                exit(-1)
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
        elif opt_val[0] == '--reg_term':
            reg_term = float(opt_val[1])
        elif opt_val[0] == '--step_size':
            step_size = float(opt_val[1])

    print '>>> input_path = %s' % str(input_path)
    print '>>> num_learners = %s' % str(num_learners)
    print '>>> num_parts = %s' % str(num_parts)
    print '>>> field_sep = %s' % str(field_sep)
    print '>>> output_path = %s' % str(output_path)
    print '>>> max_depth = %s' % str(max_depth)
    print '>>> max_features = %s' % str(max_features)
    print '>>> min_samples_leaf = %s' % str(min_samples_leaf)
    print '>>> min_samples_split = %s' % str(min_samples_split)
    print '>>> regularizer = %s' % str(regularizer)
    print '>>> niters = %s' % str(niters)
    print '>>> reg_term = %s' % str(reg_term)
    print '>>> step_size = %s' % str(step_size)

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

        for yx in p_iter:
            [ystr, xstr] = st.split(yx, field_sep, maxsplit = 1)
            yval = float(ystr)
            xvec = np.fromstring(string = xstr, sep = field_sep)

            kv_pair = (rnd.randint(1, num_learners), (yval, xvec))
            yield kv_pair

    # Map function of mapping training data to each learner (2):
    # Mapping each example to num_learners copies with their labels corrupted
    # with standard Gaussian multiplications.
    def func_pmap_rndlabeling(p_iter):
        rnd.seed()

        for yx in p_iter:
            [ystr, xstr] = st.split(yx, field_sep, maxsplit = 1)    
            yval = float(ystr)
            xvec = np.fromstring(string = xstr, sep = field_sep)

            # Emitting the training set with true labels
            yield (0, (yval, xvec))

            # Emitting the training sets with corrupted labels
            for tid in range(num_learners):
                sigma = rnd.gauss(0, 1)
                yval_rnd = yval * sigma

                kv_pair = (tid + 1, (yval_rnd, xvec))
                yield kv_pair

    # Map function of mapping training data to each learner (3):
    # Same purpose as (2), but mapping on examples.
    def func_map_rndlabeling(yx):
        rnd.seed()

        [ystr, xstr] = st.split(yx, field_sep, maxsplit = 1)
        yval = float(ystr)
        xvec = np.fromstring(string = xstr, sep = field_sep)

        # Emitting the training set with true labels
        yield (0, (yval, xvec))

        # Emitting the training sets with corrupted labels
        for tid in range(num_learners):
            sigma = rnd.gauss(0, 1)
            yval_rnd = yval * sigma

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

        learner = None
        if key != 0:
            # Train learner
            learner = tree.DecisionTreeRegressor( \
                    max_depth = max_depth, \
                    max_features = max_features, \
                    min_samples_leaf = min_samples_leaf, \
                    min_samples_split = min_samples_split \
                    )
            learner.fit(xmat, yvec)

        return (key, (learner, yvec))

    # Hypothesis sampling
    train_data_HS = sc.textFile(input_path, 1).repartition(num_parts)
    train_data_map_HS = train_data_HS.mapPartitions(func_pmap_rndlabeling).\
            combineByKey(createCombiner = lambda v : [v], \
            mergeValue = lambda c, v : c + [v], \
            mergeCombiners = lambda c1, c2 : c1 + c2 \
            )
    learner_class = train_data_map_HS.map(func_train_learner)
    learner_class_collected = learner_class.collect()

    # Map function of generating training data for coefficient fitting
    def func_map_ypredmat(yx):
        [ystr, xstr] = st.split(yx, field_sep, maxsplit = 1)
        yval = float(ystr)
        xvec = np.fromstring(string = xstr, sep = field_sep)

        values = np.zeros(num_learners + 1)
        for l in learner_class_collected:
            if l[0] != 0:
                values[l[0]] = l[1][0].predict(xvec)
            else:
                values[l[0]] = yval

        return LabeledPoint(values[0], values[1:])

    # Coefficient fitting
    coeffs = None
    train_data_CF = train_data_HS.map(func_map_ypredmat)
    if regularizer == 'l1':
        # Lasso regression
        coeffs = LassoWithSGD.train( \
                data = train_data_CF, \
                iterations = niters, \
                step = step_size, \
                regParam = reg_term \
                )
    elif regularizer == 'l2':
        # Ridge regression
        coeffs = RidgeRegressionWithSGD.train( \
                data = train_data_CF, \
                iterations = niters, \
                step = step_size, \
                regParam = reg_term \
                )
    else:
        # Least-square regression
        coeffs = LinearRegressionWithSGD.train( \
                data = train_data_CF, \
                iterations = niters, \
                step = step_size \
                )

    coeffs_enum = list(enumerate(coeffs.weights, start = 1))
    coeffs_enum.insert(0, (0, coeffs.intercept))
    coeffs_rdd = sc.parallelize(coeffs_enum)

    ensemble_learner = coeffs_rdd.join(learner_class).collect()
    
    # Save the ensemble learner
    file_ensemble_learner = open(output_path, 'w')
    pk.dump(ensemble_learner, file_ensemble_learner)
    file_ensemble_learner.close()

    # Testing code
    for l in ensemble_learner:
        print "Regressor %i: " % l[0]
        print l[1][0]
        print l[1][1][1]

    sc.stop()

if __name__ == "__main__":
    __main__()
