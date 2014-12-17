import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils

conf = SparkConf()
sc = SparkContext(conf = conf)

# Load and parse the data file into an RDD of LabeledPoint.
# Cache the data since we will use it again to compute training error.
data = MLUtils.loadLibSVMFile(sc, './input/sample_libsvm_data.txt').cache()

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={},
                                             impurity='gini', maxDepth=5, maxBins=100)

# Evaluate model on training instances and compute training error
predictions = model.predict(data.map(lambda x: x.features))
labelsAndPredictions = data.map(lambda lp: lp.label).zip(predictions)
trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(data.count())
print('Training Error = ' + str(trainErr))
print('Learned classification tree model:')
print(model)
