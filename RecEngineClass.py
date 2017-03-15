#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 00:26:59 2017

@author: alienware
"""

import os
from pyspark.mllib.recommendation import ALS
from operator import add
import itertools
import math

import csv
from io import StringIO


class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_ALL
    delimiter = ','
    quotechar = '"'
    lineterminator = '\n'
    
from pyspark import SparkContext, SQLContext

sc = SparkContext("local", "Simple App")

sqlContext = SQLContext(sc)

class RecommendationEngine:
    def __init__(self,SparkContext,rank,iterations,regularization_parameter):
        self.sc = SparkContext
        self.rank = rank
        self.iterations = iterations
        self.regularization_parameter = regularization_parameter
        self.para_combo = list(itertools.product(self.rank,self.regularization_parameter))
        pass
        
    def _train(self,rank,iterations,regularization_parameter):
        self.model = ALS.train(self.training_RDD, rank, seed=0, iterations=iterations,
                      lambda_=regularization_parameter)
    
    def _findBestCombo(self,train,test):
        self.errors = []
        for para in self.para_combo:
            model= ALS.train(train,para[0], seed=0, iterations=self.iterations,lambda_=para[1])
            true_rating = test.map(lambda d: ((d[0],d[1]),d[2]))
            prediction=model.predictAll(test.map(lambda d: (d[0],d[1]))).map(lambda d: ((d[0],d[1]),d[2]))

            true_and_pred = true_rating.join(prediction).map(lambda r:(r[0],r[1][0], r[1][1]))

            error = math.sqrt(true_and_pred.map(lambda r: (r[1]-r[2])**2).mean())
            print(para)
            print(error)
            self.errors.append(error)
        max_index = self.errors.index(max(self.errors))
        return self.para_combo[max_index]
    
    def train(self):        
        train, test = self.training_RDD.randomSplit([7, 3], seed=0)
        if len(self.para_combo)>1:
            best_para = self._findBestCombo(train,test)
        else:
            best_para = self.para_combo[0]
        self.model = ALS.train(self.training_RDD,best_para[0], seed=0, iterations=self.iterations,lambda_=best_para[1])
                
    def suggestMovies(self, users=None):
        usersTrained = self.training_RDD.map(lambda d:(d[0],1)).distinct()
        applicableUsers = usersTrained.join(users.map(lambda d: (d,1))).map(lambda d:d[0])
        if applicableUsers.count()==0:
            pass
        else:
            ratingData = self.ratingCounts.filter(lambda d:d[1]>25).cartesian(applicableUsers).map(lambda d:(d[1],d[0][0]))
            return self.predict(ratingData)
            
        
    def predict(self,X):
        return self.model.predictAll(X).map(lambda d: ((d[0],d[1]),d[2]))
    
    def createTrainingData(self, training, newUserData=None):
        if newUserData==None:
            newUserData = sc.emptyRDD()
        self.training_RDD = training.union(newUserData)
        self.ratingCounts = self.training_RDD.map(lambda data: (data[1],1)).reduceByKey(add)
        #self.X = training.map(lambda data: (data[0],data[1])).map(lambda data: (data[0],data[1]))
        #self.y = training.map(lambda data: (data[2]))
#        

datasets_path = os.path.join( 'datasets')        

ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

ratings_raw_data = sc.textFile(ratings_file)

ratings_title = ratings_raw_data.take(1)[0]

ratings_data = ratings_raw_data.filter(lambda line: line!=ratings_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0])\
.map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2])))

seed = 5
iterations = 10
regularization_parameter = [0.1]
ranks = [4]



rc1 = RecommendationEngine(sc, ranks, iterations, regularization_parameter)
rc1.createTrainingData(ratings_data)
rc1.train()

users=sc.parallelize([671,324])
ratedMovies=rc1.suggestMovies(users)


ratings_data.filter(lambda d: d[0]==671).collect()


new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,4), # Star Wars (1977)
     (0,1,3), # Toy Story (1995)
     (0,16,3), # Casino (1995)
     (0,25,4), # Leaving Las Vegas (1995)
     (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,1), # Flintstones, The (1994)
     (0,379,1), # Timecop (1994)
     (0,296,3), # Pulp Fiction (1994)
     (0,858,5) , # Godfather, The (1972)
     (0,50,4) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print ('New user ratings: %s' % new_user_ratings_RDD.take(10))

ratings_data.union(new_user_ratings_RDD)

ratings_data.take(3)
new_user_ratings_RDD.take(3)

rc1.createTrainingData(ratings_data,new_user_ratings_RDD)
rc1.train()
user = sc.parallelize([0])
rc1.suggestMovies(user).take(5)










