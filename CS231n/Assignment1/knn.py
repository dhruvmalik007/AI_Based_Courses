
# coding: utf-8

# ## Knn classifier assignment
# 
# it has two stages
# 
# 1.During training, the classifier takes the training data and simply remembers it
#   
#   
# 2.During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
#     The value of k is cross-validated
# 


import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

# rcparams set the default size of plots , mapping and other important features
plt.rcParams['figure.figsize'] =(15.0,15.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap'] = 'grey'


#loading the cifar dataset
X_train, y_train, X_test, y_test = load_CIFAR10("datasets/cifar-10-batches-py")
# checking out the parameters

print 'training data shapes:',X_train.shapes,Y_train.shape

print 'testing  data shapes:',X_test.shapes,Y_train.shape

#diving the data into ssamples for faster execution

num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
# shaping data into rows and columns 
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape

#using the pre-defined function  of the  classifier library , using the knn classifier function
classifier.train(X_train, y_train)

#  visualizing  the distance matrix: having  each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# Now finding the optimum value of k 


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.split(X_train, num_folds, axis=0)
y_train_folds = np.split(y_train, num_folds)

# A dictionary will store the accuracies for different values of k that we find
# while finding the score
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.

k_to_accuracies = {k: [] for k in k_choices}

for i in xrange(num_folds):
    # do subsetting
    X_val_curr = X_train_folds[i]
    y_val_curr = y_train_folds[i]
    X_train_curr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
    y_train_curr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

    # train classifier + compute distances for this fold
    classifier = KNearestNeighbor()
    classifier.train(X_train_curr, y_train_curr)
    dists = classifier.compute_distances_no_loops(X_val_curr)
    
    # finding accuracies on validation data for each k
    for k in xrange(1,100):
        y_val_pred_curr = classifier.predict_labels(dists, k=k)
        num_correct = np.sum(y_val_pred_curr == y_val_curr)
        accuracy = float(num_correct) / y_val_curr.shape[0]
        k_to_accuracies[k].append(accuracy)
    # printing the output according to diffrent values of k    
   for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)     
