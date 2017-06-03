#importing dependencies
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers.softmax import softmax_loss_naive #for computing the loss value
from cs231n.classifiers import Softmax # for implementing the softmax function to find the classification score
import time
import itertools #for efficient looping
# set default size of plots
plt.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# function for declaring the 
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
# Load the raw CIFAR-10 data
  cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
  X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
  
  # sampling the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]

  #reshape the image data into rows
  X_train = np.reshape(X_train, (X_train.shape[0], -1))
  X_val = np.reshape(X_val, (X_val.shape[0], -1))
  X_test = np.reshape(X_test, (X_test.shape[0], -1))
  X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
  
  #Normalizing the data around center by subtracting it with mean value of pixel values
  mean_image = np.mean(X_train, axis = 0)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  X_dev -= mean_image
  
    #  for converting into columns , adding a bias
  X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
  X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
  X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
  X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
  
  return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev
  
  #  invoking the function displaying the parameter values
  # Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print 'Train data/labels shape: ', X_train.shape , y_train.shape
print 'Validation data/labels shape: ', X_val.shape , y_val.shape
print 'Test data/labels shape: ', X_test.shape , y_test.shape
print 'dev data/labels shape: ', X_dev.shape , y_dev.shape

#generating a weight matrix and computing the loss
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
print 'loss: %f', loss


#now using the validation set to tune the hyperparameters.gets approx 0.35 classification accuracy
results = {}
best_val = -1
best_softmax = None
learning_rates = [5e-7]
regularization_strengths = [2e4]

# now iterating the loop for  finding the best possible classification score
for lr, reg in itertools.product(learning_rates, regularization_strengths):
    softmax = Softmax()
    loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=reg,
                          num_iters=num_iters, verbose=False)
    y_train_pred = softmax.predict(X_train)
    y_val_pred = softmax.predict(X_val)
    acc_train = np.mean(y_train == y_train_pred)
    acc_val = np.mean(y_val == y_val_pred)
    results[(lr, reg)] = (acc_train, acc_val)
    if acc_val > best_val:
        best_val = acc_val
        best_softmax = softmax
# display results

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved: %f' % best_val
