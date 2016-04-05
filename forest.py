#!/usr/bin/env python

import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics

import sys

mnist = sklearn.datasets.fetch_mldata("MNIST (original)")

clf_forest = sklearn.ensemble.RandomForestClassifier(30, n_jobs=100)
clf_forest.fit(mnist.data[:60000], mnist.target[:60000])

accuracy = sklearn.metrics.accuracy_score(
    mnist.target[60000:],
    clf_forest.predict(mnist.data[60000:]),
)

print 1 - accuracy
