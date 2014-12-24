import sys
import numpy as np
import pickle as pk

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark import StorageLevel

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD

model_path = ""
coeff_path = ""
coeff_data_path = ""
test_data_path = ""
fs = 'file:'
inc = 1
niters = 300
step_size = 0.00001
regularizer = None
reg_weight = 1
batch_frac = 0.1

for option in sys.argv:
    opt_val = option.split('=')
    if opt_val[0] == '--model_path':
        model_path = str(opt_val[1])
    elif opt_val[0] == '--coeff_path':
        coeff_path = str(opt_val[1])
    elif opt_val[0] == '--coeff_data_path':
        coeff_data_path = str(opt_val[1])
    elif opt_val[0] == '--test_data_path':
        test_data_path = str(opt_val[1])
    elif opt_val[0] == '--increment':
        inc = int(opt_val[1])
    elif opt_val[0] == '--niters':
        niters = int(opt_val[1])
    elif opt_val[0] == '--step_size':
        step_size = float(opt_val[1])
    elif opt_val[0] == '--regularizer':
        regularizer = str(opt_val[1])
    elif opt_val[0] == '--reg_weight':
        reg_weight = float(opt_val[1])
    elif opt_val[0] == '--batch_fraction':
        batch_frac = float(opt_val[1])

conf = SparkConf()
sc = SparkContext(conf=conf)

model_file = open(model_path, 'r')
model = pk.load(model_file)
model_file.close()
model_broadcast = sc.broadcast(model)

coeff_data_rdd = sc.pickleFile(fs + coeff_data_path).cache()
test_data_rdd = sc.pickleFile(fs + test_data_path).cache()

def func_map_pred((y, x)):
    pred = 0
    for (c, (k, l)) in zip(coeff_used_list, model_broadcast.value[0:num_learners_used]):
        pred += c * l.predict(x)

    return (y, pred)

num_learners = len(model)
num_learners_used = inc
learning_curve = []
while num_learners_used <= num_learners:
    coeff_data_rdd_used = coeff_data_rdd.map( \
            lambda lp : LabeledPoint(lp.label, lp.features[0:num_learners_used]))
    
    coeff_used = LinearRegressionWithSGD.train( \
            data = coeff_data_rdd_used, \
            iterations = niters, \
            step = step_size, \
            regType = regularizer, \
            regParam = reg_weight, \
            miniBatchFraction = batch_frac, \
            intercept = False
            )

    coeff_used_list = coeff_used.weights

    learning_curve.append(test_data_rdd \
            .map(func_map_pred) \
            .map(lambda (y, pred) : (y - pred) * (y - pred) / 2) \
            .mean() \
            )

    num_learners_used += inc

file = open('/filer/tmp1/yw298/spark/output/learning_curve', 'w')
pk.dump(learning_curve, file)
file.close()
