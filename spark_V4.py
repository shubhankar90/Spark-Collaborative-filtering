#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:23:54 2017

@author: alienware
"""

import os

import csv
from io import StringIO
from operator import add

class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_ALL
    delimiter = ','
    quotechar = '"'
    lineterminator = '\n'



datasets_path = os.path.join( 'datasets')

from pyspark import SparkContext, SQLContext

sc = SparkContext("local", "Simple App")

sqlContext = SQLContext(sc)


ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

ratings_raw_data = sc.textFile(ratings_file)

ratings_title = ratings_raw_data.take(1)[0]

ratings_data = ratings_raw_data.filter(lambda line: line!=ratings_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0])\
.map(lambda tokens: (tokens[0],tokens[1],tokens[2]))


training_RDD, validation_RDD, test_RDD = \
ratings_data.randomSplit([6, 2, 2], seed=0)
    
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank
        
        





