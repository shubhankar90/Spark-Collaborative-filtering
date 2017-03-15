#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:50:42 2017

@author: alienware
"""
import os
import urllib
import itertools
import pandas

from pyspark import SparkContext, SQLContext
from pyspark.sql import Row

import csv
from io import StringIO


class MyDialect(csv.Dialect):
    strict = True
    skipinitialspace = True
    quoting = csv.QUOTE_ALL
    delimiter = ','
    quotechar = '"'
    lineterminator = '\n'

    

sc = SparkContext("local", "Simple App")

sqlContext = SQLContext(sc)


datasets_path = os.path.join( 'datasets')

ratings_file = sc.textFile(os.path.join(datasets_path, 'ml-latest', 'ratings.csv'))
movies_file = sc.textFile(os.path.join(datasets_path, 'ml-latest', 'movies.csv'))
links_file = sc.textFile(os.path.join(datasets_path, 'ml-latest', 'links.csv'))
tags_file = sc.textFile(os.path.join(datasets_path, 'ml-latest', 'tags.csv'))

ratings_sample = ratings_file.take(5)
movies_sample = movies_file.take(5)
links_sample = links_file.take(5)
tags_sample = tags_file.take(40)

ratings_title = ratings_file.take(1)[0]
movies_title = movies_file.take(1)[0]
links_title = links_file.take(1)[0]
tags_title = tags_file.take(1)[0]


ratings_data = ratings_file.filter(lambda line: line!=ratings_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0]).map(lambda tokens: (tokens[0],tokens[1],tokens[2]))

movies_data = movies_file.filter(lambda line: line!=movies_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0]).map(lambda tokens: (tokens[0],tokens[1],tokens[2]))

links_data = links_file.filter(lambda line: line!=links_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0]).map(lambda tokens: (tokens[0],tokens[1],tokens[2]))

tags_data = tags_file.filter(lambda line: line!=tags_title)\
.map(lambda line: [item for item in csv.reader(StringIO(line),MyDialect())][0]).map(lambda tokens: (tokens[0],tokens[1],tokens[2]))

movie_genre = movies_data.flatMap(lambda line: [(line[0],item) for item in line[2].split('|')])

movie_genre_title = 'movieId,genre'

movie_genre_sample = movie_genre.take(5)

from operator import add
tags_data.map(lambda data:((data[0],data[1]),1)).reduceByKey(add).filter(lambda data:data[1]>1).take(10)
tags_data.filter(lambda data: (data[0]=='630')&(data[1]=='260')).collect()

tags_data_grouped = tags_data.groupBy(lambda data:data[1]).map(lambda data: (data[0],[item[2] for item in data[1]]))

              




tags_data_grouped_df = sqlContext.createDataFrame(tags_data_grouped.map(lambda data: Row(MovieID=data[0],Tags_comb = data[1])))

movie_genre_df = sqlContext.createDataFrame(movie_genre.map(lambda data: Row(MovieID=data[0],Genre = data[1])))

genre_tags = movie_genre_df.join(tags_data_grouped_df,movie_genre_df.MovieID==tags_data_grouped_df.MovieID,"inner").drop(movie_genre_df.MovieID)

def uniqueT(data):
    return list(set(itertools.chain.from_iterable(data)))

genre_tags=genre_tags.drop('MovieID').rdd.groupBy(lambda data:data[0]).map(lambda data: (data[0],[item[1] for item in data[1]])).map(lambda data: (data[0], uniqueT(data[1])))

genre_tags_pdf=sqlContext.createDataFrame(genre_tags).toPandas()



                   
                          
                                       
genre_tags_pdf.as_matrix()[13,:]
                                       
genre_tags_pdf._1
                      


movie_ratings_df = sqlContext.createDataFrame(ratings_data.map(lambda data:((data[0],data[2]),1)).reduceByKey(add).map(lambda data: (data[0][0],data[0][1],data[1])).map(lambda data: Row(MovieID=data[0],Rating=data[1],RatingCount=data[2])))

genre_rating_pdf = sqlContext.createDataFrame(movie_genre_df.join(movie_ratings_df,movie_genre_df.MovieID==movie_ratings_df.MovieID,"inner").drop(movie_genre_df.MovieID).rdd.map(lambda data:((data[0],data[2]),data[3])).reduceByKey(add).map(lambda data: (data[0][0],data[0][1],data[1]))).toPandas()

genre_rating_pdf[genre_rating_pdf._1=='Mystery']
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       