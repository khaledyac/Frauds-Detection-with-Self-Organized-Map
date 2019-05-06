# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:45:03 2019

@author: K6433702
"""
#%reset -f
#%clear

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
print('The dimension of the dataset is {0}'.format(dataset.shape))

# Data preparing
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
"""we splited our data sets into x and y. But be careful we did not do that because we're doing some supervised learning, we're not trying to make a model
that will predict zero or one in the end. We're just doing this to make the distinction in the end between the customers who were approved
and the customers who were not approved. You will see that when we train our self organizing map we will only use x because we are doing some
unsupervised deep learning, that means that no dependent variable is considered."""

# Features scaling
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range=(0,1))
X = mm.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10,y = 10,input_len = 15,sigma = 1.0,learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X,num_iteration = 100)

# visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the Frauds
mappings  = som.win_map(X)
frauds = np.concatenate((mappings[(8,9)], mappings[(8,3)],mappings[(8,2)]),axis=0)
frauds = mm.inverse_transform(frauds)
